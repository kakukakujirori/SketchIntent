from typing import Any
import random
import numpy as np
import sketchgraphs.data as datalib

from .constants import (
    PrimitiveToken, PrimitiveAttrToken, ConstraintToken, ConstraintAttrToken,
    PRIMITIVE_VAL_OFFSET, CONSTRAINT_VAL_OFFSET, SpecialToken
)
from .context import SketchContext


class SketchTokenizer:
    """
    Tokenizer/Detokenizer for SketchGraphs data.
    """
    def __init__(self, context: SketchContext):
        self.context = context

        self.max_token_length = self.context.max_token_length
        self.align_constraint_token_length_to = self.context.align_constraint_token_length_to or -1
        self.tail_padding_token = SpecialToken.Stop if self.context.use_stop_padding else SpecialToken.Pad
        self.constraint_order = self.context.constraint_order

    @staticmethod
    def quantize_params(params: np.ndarray, n_bins: int, min_val: float = -0.5, max_val: float = 0.5):
        """Convert params in [MIN_VAL, MAX_VAL] to discrete values in [0, n_bins-1]."""
        if (params < min_val).any() or (params > max_val).any():
            raise ValueError("Parameters must be in [%f, %f]. Got [%f, %f]."
            % (min_val, max_val, np.min(params), np.max(params)))
        params_quantized = (params - min_val) / (max_val - min_val) * n_bins
        params_quantized = params_quantized.astype('int32')
        # Handle max_val edge case
        params_quantized[params_quantized == n_bins] -= 1
        return params_quantized

    @staticmethod
    def dequantize_params(params: np.ndarray, n_bins: int, min_val: float = -0.5, max_val: float = 0.5):
        """Convert quantized parameters to floats in range [MIN_VAL, MAX_VAL]."""
        if ((params < 0).any() or (params >= n_bins).any()
            or not np.issubdtype(params.dtype, np.integer)):
            raise ValueError("Invalid quantized params")
        params = params.astype('float32') + 0.5  # center of each bin
        params = params / n_bins * (max_val - min_val) + min_val
        return params

    def pad_or_truncate(self, arr: np.ndarray, length: int | None = None) -> np.ndarray:
        """Pad or truncate array to fixed length."""
        if length is None:
            return arr
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), constant_values=self.tail_padding_token)

    def tokenize_primitives(self, sketch: datalib.Sketch) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """
        Tokenize primitives into 'val', 'coord', 'pos' arrays.
        Returns tokens and a primitive_tokenpos for constraint reference resolution.
        """
        val_tokens = [PrimitiveToken.Start]
        attr_tokens = [PrimitiveAttrToken.Type]
        pos_tokens = [PrimitiveToken.Start]

        primitive_tokenpos: dict[str, int] = {} # Entity ID -> Token Index (or Subnode -> Index)

        entities = list(sketch.entities.values())
        val_offset = PRIMITIVE_VAL_OFFSET
        num_bins = self.context.num_quantization_bins

        for ent_idx, ent in enumerate(entities):
            handler = self.context.get_primitive_handler(type(ent).__name__)
            if not handler: continue

            # 1. Type
            pos_idx = pos_tokens[-1] + 1
            ent_start_pos = len(val_tokens) # Position of Type Token

            val_tokens.append(handler.token_id)
            attr_tokens.append(PrimitiveAttrToken.Type)
            pos_tokens.append(pos_idx)

            # 2. Params
            vals, p_tokens = handler.encode(ent)
            vals = np.clip(vals, -0.5, 0.5)  # Arc.Center can be outside [-0.5, 0.5]^2
            q_vals = self.quantize_params(vals, num_bins)

            val_tokens.extend(q_vals + val_offset)
            attr_tokens.extend(p_tokens)
            pos_tokens.extend([pos_idx] * len(q_vals))

            # 3. Construction
            is_constr = int(ent.isConstruction)
            val_tokens.append(val_offset + num_bins + is_constr)
            attr_tokens.append(PrimitiveAttrToken.Construction)
            pos_tokens.append(pos_idx)

            # 4. Register in primitive_tokenpos
            ent_id = ent.entityId
            subnode_offsets = handler.get_subnode_offset()
            if len(val_tokens) <= self.max_token_length:
                primitive_tokenpos[ent_id] = ent_start_pos
                for sub_name, sub_offset in subnode_offsets.items():
                    primitive_tokenpos[f"{ent_id}.{sub_name}"] = ent_start_pos + sub_offset
            else:
                # Truncated primitives are assigned -1 in primitive_tokenpos
                primitive_tokenpos[ent_id] = -1
                for sub_name, sub_offset in subnode_offsets.items():
                    primitive_tokenpos[f"{ent_id}.{sub_name}"] = -1

        # Stop Token
        val_tokens.append(PrimitiveToken.Stop)
        attr_tokens.append(PrimitiveAttrToken.Type)
        pos_tokens.append(pos_tokens[-1] + 1)

        tokens = {
            'p_val': self.pad_or_truncate(np.array(val_tokens, dtype=np.int64), self.max_token_length),
            'p_attr': self.pad_or_truncate(np.array(attr_tokens, dtype=np.int64), self.max_token_length),
            'p_pos': self.pad_or_truncate(np.array(pos_tokens, dtype=np.int64), self.max_token_length),
            'p_cnt': np.array([pos_tokens[-1] - pos_tokens[0] - 1], dtype=np.int64),
        }
        return tokens, primitive_tokenpos

    def tokenize_constraints(
        self,
        sketch: datalib.Sketch,
        primitive_tokenpos: dict[str, int],
        allow_invalid_constraints: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Tokenize constraints into separate 'val', 'coord', 'pos' arrays.
        Uses primitive_tokenpos from primitives to resolve references.

        Args:
            sketch: The sketch object.
            primitive_tokenpos: dict mapping entity ID -> token position.
            allow_invalid_constraints: If True, constraints with invalid references are skipped.
        """
        # Phase 1: Parse and Filter
        parsed_constraints = []

        for constr in sketch.constraints.values():
            constraint_type = constr.constraint_type.name

            # 0. Get generic handler
            handler = self.context.get_constraint_handler(constraint_type)
            if not handler: continue  # Skip unknown types

            # 1. Delegate to handler.encode
            # This handles type refinement (Distance->DistanceX), ref parsing, filtering, and sorting
            constr_id, ref_tokens, params, p_tokens = handler.encode(constr, primitive_tokenpos)

            # Constraints exceeding max_token_length are discarded
            if self.max_token_length is not None:
                if any(t_pos >= self.max_token_length for t_pos in ref_tokens):
                    continue

            if handler.validate_refs(tuple(ref_tokens)):
                parsed_constraints.append((constr_id, ref_tokens, params, p_tokens))
            elif len(ref_tokens) > 0 and not allow_invalid_constraints:
                # NOTE: Handlers don't check refs in encode to allow invalid constraints
                # NOTE: If ref_tokens=[], it may be because the referring primitives are external or truncated.
                raise ValueError(f"Invalid refs in constraint: {constr}")

        # Sort constraints based on constraint_order
        if self.constraint_order == "sort_by_ref_id":
            parsed_constraints.sort(key=lambda x: min(x[1]) if x[1] else 0)
        elif self.constraint_order == "random_shuffle":
            random.shuffle(parsed_constraints)
        elif self.constraint_order == "original":
            pass
        else:
            raise NotImplementedError(f"Unknown constraint_order: {self.constraint_order}")

        # Phase 2: Emit Tokens
        val_tokens = [ConstraintToken.Start]
        attr_tokens = [ConstraintAttrToken.Type]
        pos_tokens = [ConstraintToken.Start]
        val_offset = CONSTRAINT_VAL_OFFSET

        for constr_id, ref_tokens, params, p_tokens in parsed_constraints:
            # Type Token
            pos_idx = pos_tokens[-1] + 1
            val_tokens.append(constr_id)
            attr_tokens.append(ConstraintAttrToken.Type)
            pos_tokens.append(pos_idx)

            # Reference Tokens
            for t_pos in ref_tokens:
                # Uniquification adds offset to pointers
                val_tokens.append(t_pos + val_offset)
                attr_tokens.append(ConstraintAttrToken.Reference)
                pos_tokens.append(pos_idx)

            # NOTE: Parameters are skipped

            # Fixed length padding (InnerPad)
            if self.align_constraint_token_length_to > 0:
                num_refs = len(ref_tokens)
                max_refs = self.align_constraint_token_length_to - 1
                if max_refs - num_refs < 0:
                    raise ValueError(f"Constraint has {num_refs} refs, exceeds max {max_refs} for fixed format")
                for _ in range(max_refs - num_refs):
                    val_tokens.append(ConstraintToken.InnerPad)
                    attr_tokens.append(ConstraintAttrToken.InnerPad)
                    pos_tokens.append(pos_idx)

        # Stop Token
        val_tokens.append(ConstraintToken.Stop)
        attr_tokens.append(ConstraintAttrToken.Type)
        pos_tokens.append(pos_tokens[-1] + 1)

        return {
            'c_val': self.pad_or_truncate(np.array(val_tokens, dtype=np.int64), self.max_token_length),
            'c_attr': self.pad_or_truncate(np.array(attr_tokens, dtype=np.int64), self.max_token_length),
            'c_pos': self.pad_or_truncate(np.array(pos_tokens, dtype=np.int64), self.max_token_length),
            'c_cnt': np.array([pos_tokens[-1] - pos_tokens[0] - 1], dtype=np.int64),
        }

    def tokenize(self, sketch: datalib.Sketch, allow_invalid_constraints: bool = False) -> dict[str, np.ndarray]:
        """
        Tokenize a sketch into primitives and constraints.
        """
        try:
            prim_tokens, primitive_tokenpos = self.tokenize_primitives(sketch)
            constr_tokens = self.tokenize_constraints(sketch, primitive_tokenpos, allow_invalid_constraints)
        except ValueError as e:
            print(f"=== ERROR: Sketch Tokenization FAILED ===")
            print(f"  {sketch}")
            print(f"  Entities:")
            for eid, ent in sketch.entities.items():
                print(f"    {ent}")
            print(f"  Constraints:")
            for cid, constr in sketch.constraints.items():
                params_str = ", ".join(f"{type(p).__name__}({p.parameterId}={getattr(p, 'value', '?')})" for p in constr.parameters)
                print(f"    {constr.constraint_type.name} [{params_str}]")
            print(f"  >>> FAILED: {e}")
            raise e
        return prim_tokens | constr_tokens

    def detokenize_primitives(self, val_tokens: np.ndarray) -> tuple[dict[str, datalib.Entity], dict[int, str]]:
        """
        Detokenize primitives and build tokenpos_to_primitive.
        """
        entities = []
        tokenpos_to_primitive = {}

        current_ent_idx = 0
        num_bins = self.context.num_quantization_bins
        val_offset = PRIMITIVE_VAL_OFFSET

        # Skip Start
        idx = 1 if val_tokens[0] == SpecialToken.Start else 0

        while idx < len(val_tokens):
            tid = val_tokens[idx]
            if tid == SpecialToken.Stop or tid == SpecialToken.Pad:
                break

            handler = self.context._token_id_to_primitive_handler.get(tid)
            if not handler:
                raise ValueError(f"Invalid token id: {tid}")

            # Determine entity/param/construction position
            n_params = handler.n_params
            ent_start_pos = idx + 0
            param_start_pos = idx + 1
            construction_pos = idx + 1 + n_params
            if construction_pos >= len(val_tokens): break

            # Params
            param_bins = val_tokens[param_start_pos : construction_pos] - val_offset
            param_bins = np.clip(param_bins, 0, num_bins - 1)
            vals = self.dequantize_params(param_bins, num_bins)

            # Construction
            constr_token = val_tokens[construction_pos]
            is_constr = (constr_token - (val_offset + num_bins)) > 0

            # Create
            ent_id = str(current_ent_idx)
            entity = handler.decode(vals, entity_id=ent_id)
            entity.isConstruction = bool(is_constr)
            entities.append(entity)

            # Map subnodes
            tokenpos_to_primitive[ent_start_pos] = ent_id
            sub_offsets = handler.get_subnode_offset()
            for sub_name, offset in sub_offsets.items():
                abs_idx = ent_start_pos + offset
                tokenpos_to_primitive[abs_idx] = f"{ent_id}.{sub_name}"

            # Increment
            current_ent_idx += 1
            idx += 1 + n_params + 1

        return {e.entityId: e for e in entities}, tokenpos_to_primitive

    def parse_constraint_tokens(self, val_tokens: np.ndarray) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
        """Parse constraint token sequence into lightweight (type_id, ref_positions) tuples.

        Handles both variable-length and fixed-block modes based on
        ``self.align_constraint_token_length_to``. The block mode is more robust
        to noisy/corrupted model output since it self-corrects at each block
        boundary.

        Args:
            val_tokens: 1-D array of constraint value tokens (may start with
                Start token, may end with Stop/Pad).

        Returns:
            List of ``(constraint_type_token_id, (sorted_ref_positions...), corresponding original token sequence)``.
        """
        val_offset = CONSTRAINT_VAL_OFFSET
        tokens = val_tokens.tolist() if hasattr(val_tokens, 'tolist') else list(val_tokens)

        # Skip Start token
        idx = 1 if tokens and tokens[0] == SpecialToken.Start else 0
        tokens = tokens[idx:]

        parsed: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

        if self.align_constraint_token_length_to is None or self.align_constraint_token_length_to <= 0:
            # --- Variable-length mode ---
            block = []

            for tok in tokens:
                tok = int(tok)

                if tok == SpecialToken.Pad:
                    continue
                if tok == SpecialToken.Stop:
                    break
                if tok == ConstraintToken.InnerPad:
                    continue

                if tok < val_offset:
                    # This is a type token (or other special) — save previous
                    if block is not None:
                        current_type = block[0]
                        current_refs = tuple(sorted([x - val_offset for x in block[1:]]))
                        parsed.append((current_type, current_refs, tuple(block)))
                    block = [tok]
                else:
                    # Reference token
                    block.append(tok - val_offset)

            # Don't forget the last constraint
            if block:
                current_type = block[0]
                current_refs = tuple(int(b) - val_offset for b in block[1:])
                parsed.append((current_type, current_refs, tuple(block)))

        else:
            # --- Fixed block mode ---
            block_size = self.align_constraint_token_length_to
            block_idx = 0

            while block_idx + block_size <= len(tokens):
                block = tokens[block_idx:block_idx + block_size]
                block_idx += block_size

                type_tok = int(block[0])

                if type_tok == SpecialToken.Stop:
                    break

                # Skip invalid blocks (Pad, Stop, Mask, InnerPad, or non-type token)
                if type_tok in (SpecialToken.Pad, SpecialToken.Mask, ConstraintToken.InnerPad):
                    continue
                if type_tok >= val_offset:
                    continue

                # Collect valid ref tokens (strip InnerPad from tail, then validate)
                ref_block = list(block[1:])
                while ref_block and int(ref_block[-1]) == ConstraintToken.InnerPad:
                    ref_block.pop()

                # All remaining tokens must be valid ref tokens (>= val_offset)
                if not all(int(b) >= val_offset for b in ref_block):
                    continue

                refs = tuple(int(b) - val_offset for b in ref_block)
                parsed.append((type_tok, refs, tuple(block)))

        return parsed

    def detokenize_constraints(
        self,
        val_tokens: np.ndarray,
        tokenpos_to_primitive: dict[int, str],
        allow_invalid_constraints: bool = False
    ) -> tuple[dict[str, datalib.Constraint], list[tuple[int, ...]]]:
        """Detokenize constraints using the map from primitives.

        Uses :meth:`parse_constraint_tokens` for robust token parsing, then
        delegates to each constraint handler's ``decode`` method.
        """
        constraints = {}
        invalid_token_blocks = []
        parsed = self.parse_constraint_tokens(val_tokens)

        for type_id, ref_positions, token_block in parsed:
            try:
                # Look up handler by constraint type name
                c_name = ConstraintToken(type_id).name
                handler = self.context.get_constraint_handler(c_name)
                assert handler is not None

                # Validation check (NOTE: handlers don't check refs in decode to allow invalid constraints)
                if not allow_invalid_constraints and handler.validate_refs(ref_positions):
                    raise ValueError(f"Invalid refs for constraint {c_name}: {ref_positions}")

                # Pass empty array for params as they are not tokenized
                constr = handler.decode(ref_positions, np.array([]), tokenpos_to_primitive)
                assert constr is not None
                unique_id = f"c_{len(constraints)}"
                constr = constr._replace(identifier=unique_id)
                constraints[unique_id] = constr
            except Exception as e:
                if allow_invalid_constraints:
                    invalid_token_blocks.append(token_block)
                    continue
                else:
                    raise e

        return constraints, invalid_token_blocks

    def detokenize(self, tokens: dict[str, np.ndarray], allow_invalid_constraints: bool = False) -> tuple[datalib.Sketch, list[tuple[int, ...]]]:
        """
        Detokenize separate primitive and constraint streams into a Sketch object.
        """
        entities, tokenpos_to_primitive = self.detokenize_primitives(tokens['p_val'])
        constraints, invalid_token_blocks = self.detokenize_constraints(tokens['c_val'], tokenpos_to_primitive, allow_invalid_constraints)

        return datalib.Sketch(entities=entities, constraints=constraints), invalid_token_blocks
