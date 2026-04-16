# 将1个pdb结构文件量化为1个 节点+边接触事件

# 1.
# 简化版，主要使用的SIFTS是通过PDBe api映射

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import httpx
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser


class StructureEventQuantizer:
    """
    结构事件量化器。

    功能概述
    --------
    1) 输入 1 个 PDB ID，自动下载或复用本地 mmCIF 文件。
    2) 解析每个建模残基（当前使用 CA 坐标）并构建节点表。
    3) 通过 PDBe SIFTS 将结构残基编号映射到 UniProt 坐标。
    4) 构建边表（残基对距离），可选阈值过滤。
    """

    def __init__(
        self,
        root_dir: str = "/data2",
        cif_subdir: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.cif_dir = self.root_dir / cif_subdir if cif_subdir else self.root_dir
        self.cif_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @staticmethod
    def _norm_pdb_id(pdb_id: str) -> str:
        return str(pdb_id).strip().upper()

    def _find_local_cif(self, pdb_id: str) -> Optional[Path]:
        pdb = self._norm_pdb_id(pdb_id)
        candidates = [
            self.cif_dir / f"{pdb}.cif",
            self.cif_dir / f"{pdb.lower()}.cif",
            self.cif_dir / f"{pdb.lower()}.mmcif",
            self.cif_dir / f"{pdb}.mmcif",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def download_cif(self, pdb_id: str, overwrite: bool = False) -> Path:
        """下载 RCSB mmCIF 文件到本地。"""
        pdb = self._norm_pdb_id(pdb_id)
        out_path = self.cif_dir / f"{pdb.lower()}.cif"

        if out_path.exists() and not overwrite:
            return out_path

        url = f"https://files.rcsb.org/download/{pdb.lower()}.cif"
        resp = httpx.get(url, timeout=self.timeout)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return out_path

    def _ensure_cif(self, pdb_id: str, download_if_missing: bool = True) -> Path:
        local = self._find_local_cif(pdb_id)
        if local is not None:
            return local
        if not download_if_missing:
            raise FileNotFoundError(f"未找到本地 mmCIF 文件: {pdb_id}")
        return self.download_cif(pdb_id)

    def fetch_sifts_ranges(self, pdb_id: str) -> pd.DataFrame:
        """
        获取 PDBe SIFTS 区间映射。

        返回字段
        --------
        - chain_id / struct_asym_id / auth_asym_id
        - pdb_start / pdb_end
        - uniprot_start / uniprot_end
        - uniprot_id
        """
        pdb = self._norm_pdb_id(pdb_id)
        pdb_norm = pdb.lower()
        urls = [
            f"https://www.ebi.ac.uk/pdbe/api/v2/mappings/uniprot/{pdb_norm}",
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_norm}",
        ]

        payload = None
        last_error: Optional[Exception] = None
        for url in urls:
            try:
                resp = httpx.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if pdb_norm in data:
                    payload = data[pdb_norm]
                    break
            except Exception as exc:  # pragma: no cover - 网络波动分支
                last_error = exc

        if payload is None:
            raise RuntimeError(f"无法获取 SIFTS 映射: {pdb}, error={last_error}")

        rows: List[Dict[str, Any]] = []
        for uniprot_id, block in payload.get("UniProt", {}).items():
            for m in block.get("mappings", []):
                start = m.get("start", {})
                end = m.get("end", {})

                pdb_start = start.get("author_residue_number", start.get("residue_number"))
                pdb_end = end.get("author_residue_number", end.get("residue_number"))
                if pdb_start is None or pdb_end is None:
                    continue

                rows.append(
                    {
                        "pdb_id": pdb,
                        "uniprot_id": str(uniprot_id),
                        "entity_id": m.get("entity_id"),
                        "chain_id": m.get("auth_asym_id") or m.get("chain_id") or m.get("struct_asym_id"),
                        "struct_asym_id": m.get("struct_asym_id"),
                        "auth_asym_id": m.get("auth_asym_id", m.get("chain_id")),
                        "pdb_start": int(pdb_start),
                        "pdb_end": int(pdb_end),
                        "uniprot_start": int(m["unp_start"]),
                        "uniprot_end": int(m["unp_end"]),
                    }
                )

        sifts_df = pd.DataFrame(rows)
        if not sifts_df.empty:
            sifts_df = sifts_df.sort_values(["chain_id", "pdb_start", "pdb_end"]).reset_index(drop=True)
        return sifts_df

    def parse_mmcif_ca_residues(
        self,
        cif_path: str,
        pdb_id: Optional[str] = None,
        model_index: int = 0,
    ) -> pd.DataFrame:
        """
        解析 mmCIF 中的 CA 残基，构建节点基础表（未带 UniProt 映射）。

        节点层索引设计
        ------------
        - global_node_idx0/global_node_idx1: 全结构连续索引（用于全图）。
        - chain_node_idx0/chain_node_idx1: 链内连续索引（用于链内切片）。
        - auth_seq_id: 结构原始残基编号（可能不连续）。
        """
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_path)
        models = list(structure.get_models())
        if model_index >= len(models):
            raise IndexError(f"model_index 越界: {model_index}, total={len(models)}")
        model = models[model_index]

        inferred_id = (pdb_id or structure.header.get("idcode") or Path(cif_path).stem[:4]).upper()

        rows: List[Dict[str, Any]] = []
        for chain in model:
            for residue in chain:
                # 仅保留标准残基且存在 CA 原子
                if not residue.has_id("CA"):
                    continue
                hetflag, resseq, icode = residue.id
                if isinstance(hetflag, str) and hetflag.strip():
                    continue

                coord = residue["CA"].get_coord()
                rows.append(
                    {
                        "pdb_id": inferred_id,
                        "chain_id": str(chain.id),
                        "residue_name": str(residue.get_resname()),
                        "auth_seq_id": int(resseq),
                        "insertion_code": "" if icode in (" ", "") else str(icode),
                        "ca_x": float(coord[0]),
                        "ca_y": float(coord[1]),
                        "ca_z": float(coord[2]),
                    }
                )

        node_df = pd.DataFrame(rows)
        if node_df.empty:
            return node_df

        node_df = node_df.sort_values(["chain_id", "auth_seq_id", "insertion_code"]).reset_index(drop=True)
        node_df["chain_node_idx0"] = node_df.groupby("chain_id").cumcount().astype(int)
        node_df["chain_node_idx1"] = node_df["chain_node_idx0"] + 1
        node_df["global_node_idx0"] = np.arange(len(node_df), dtype=int)
        node_df["global_node_idx1"] = node_df["global_node_idx0"] + 1
        node_df["CA_coord"] = node_df[["ca_x", "ca_y", "ca_z"]].values.tolist()
        return node_df

    @staticmethod
    def _build_chain_segment_dict(sifts_df: pd.DataFrame) -> Dict[str, List[Tuple[int, int, int, int, str]]]:
        segment_map: Dict[str, List[Tuple[int, int, int, int, str]]] = {}
        if sifts_df.empty:
            return segment_map

        for _, row in sifts_df.iterrows():
            seg = (
                int(row["pdb_start"]),
                int(row["pdb_end"]),
                int(row["uniprot_start"]),
                int(row["uniprot_end"]),
                str(row["uniprot_id"]),
            )
            # 兼容多种链标识写法
            for cid in {row.get("chain_id"), row.get("struct_asym_id"), row.get("auth_asym_id")}:
                if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                    continue
                key = str(cid).strip()
                if not key:
                    continue
                segment_map.setdefault(key, []).append(seg)

        for k in segment_map:
            segment_map[k].sort(key=lambda x: (x[0], x[1]))
        return segment_map

    @staticmethod
    def _map_to_uniprot(
        chain_id: str,
        auth_seq_id: int,
        segment_map: Dict[str, List[Tuple[int, int, int, int, str]]],
    ) -> Tuple[Optional[str], Optional[int]]:
        for pdb_start, pdb_end, unp_start, unp_end, uniprot_id in segment_map.get(chain_id, []):
            if pdb_start <= auth_seq_id <= pdb_end:
                unp = unp_start + (auth_seq_id - pdb_start)
                if unp <= unp_end:
                    return uniprot_id, int(unp)
        return None, None

    def attach_uniprot_mapping(self, node_df: pd.DataFrame, sifts_df: pd.DataFrame) -> pd.DataFrame:
        """把 UniProt 映射写入节点表。"""
        if node_df.empty:
            return node_df.copy()

        out = node_df.copy()
        seg_map = self._build_chain_segment_dict(sifts_df)

        mapped_uid: List[Optional[str]] = []
        mapped_pos: List[Optional[int]] = []

        for _, row in out.iterrows():
            uid, up = self._map_to_uniprot(str(row["chain_id"]), int(row["auth_seq_id"]), seg_map)
            mapped_uid.append(uid)
            mapped_pos.append(up)

        out["uniprot_id"] = mapped_uid
        out["uniprot_pos"] = pd.array(mapped_pos, dtype="Int64")
        return out

    @staticmethod
    def _serialize_df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].map(lambda x: isinstance(x, (list, tuple, dict))).any():
                out[col] = out[col].map(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, tuple, dict)) else x
                )
        return out

    def build_edge_table(
        self,
        node_df: pd.DataFrame,
        inter_chain_only: bool = False,
        distance_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        构建边表（残基对）。

        参数说明
        --------
        - inter_chain_only=True: 仅保留链间边。
        - distance_cutoff=None: 不限制距离，保留全部候选边。
        """
        if node_df.empty:
            return pd.DataFrame()

        by_chain = {
            cid: g.sort_values("chain_node_idx0").reset_index(drop=True)
            for cid, g in node_df.groupby("chain_id")
        }
        chain_ids = sorted(by_chain.keys())

        if inter_chain_only:
            pairs = list(combinations(chain_ids, 2))
        else:
            pairs = []
            for i, ci in enumerate(chain_ids):
                for cj in chain_ids[i:]:
                    pairs.append((ci, cj))

        edge_parts: List[pd.DataFrame] = []

        for ci, cj in pairs:
            gi = by_chain[ci]
            gj = by_chain[cj]

            xi = gi[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=np.float32)
            xj = gj[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=np.float32)
            dmat = np.linalg.norm(xi[:, None, :] - xj[None, :, :], axis=2)

            if ci == cj:
                ii, jj = np.triu_indices(len(gi), k=1)
                dist = dmat[ii, jj]
            else:
                ii, jj = np.indices(dmat.shape)
                ii = ii.ravel()
                jj = jj.ravel()
                dist = dmat.ravel()

            if distance_cutoff is not None:
                m = dist <= float(distance_cutoff)
                ii = ii[m]
                jj = jj[m]
                dist = dist[m]

            if len(dist) == 0:
                continue

            gi_chain_idx = gi["chain_node_idx0"].to_numpy(dtype=int)
            gi_global_idx = gi["global_node_idx0"].to_numpy(dtype=int)
            gi_auth = gi["auth_seq_id"].to_numpy(dtype=int)
            gi_uid = gi["uniprot_id"].to_numpy()
            gi_up = gi["uniprot_pos"].to_numpy()

            gj_chain_idx = gj["chain_node_idx0"].to_numpy(dtype=int)
            gj_global_idx = gj["global_node_idx0"].to_numpy(dtype=int)
            gj_auth = gj["auth_seq_id"].to_numpy(dtype=int)
            gj_uid = gj["uniprot_id"].to_numpy()
            gj_up = gj["uniprot_pos"].to_numpy()

            part = pd.DataFrame(
                {
                    "pdb_id": np.repeat(str(gi["pdb_id"].iat[0]), len(dist)),
                    "chain_i": np.repeat(ci, len(dist)),
                    "chain_node_idx0_i": gi_chain_idx[ii],
                    "global_node_idx0_i": gi_global_idx[ii],
                    "auth_seq_id_i": gi_auth[ii],
                    "uniprot_id_i": gi_uid[ii],
                    "uniprot_pos_i": gi_up[ii],
                    "chain_j": np.repeat(cj, len(dist)),
                    "chain_node_idx0_j": gj_chain_idx[jj],
                    "global_node_idx0_j": gj_global_idx[jj],
                    "auth_seq_id_j": gj_auth[jj],
                    "uniprot_id_j": gj_uid[jj],
                    "uniprot_pos_j": gj_up[jj],
                    "distance": dist.astype(np.float32),
                }
            )
            edge_parts.append(part)

        if not edge_parts:
            return pd.DataFrame(
                columns=[
                    "pdb_id",
                    "chain_i",
                    "chain_node_idx0_i",
                    "global_node_idx0_i",
                    "auth_seq_id_i",
                    "uniprot_id_i",
                    "uniprot_pos_i",
                    "chain_j",
                    "chain_node_idx0_j",
                    "global_node_idx0_j",
                    "auth_seq_id_j",
                    "uniprot_id_j",
                    "uniprot_pos_j",
                    "distance",
                ]
            )

        edge_df = pd.concat(edge_parts, ignore_index=True)
        return edge_df

    def quantize_from_pdb_id(
        self,
        pdb_id: str,
        model_index: int = 0,
        inter_chain_only: bool = False,
        distance_cutoff: Optional[float] = None,
        download_if_missing: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        主入口：给定 PDB ID，输出节点表与边表。

        返回
        ----
        - node_table: 每行一个建模残基节点。
        - edge_table: 每行一条边（残基对距离）。
        - sifts_ranges: SIFTS 区间映射明细。
        """
        pdb = self._norm_pdb_id(pdb_id)
        cif_path = self._ensure_cif(pdb, download_if_missing=download_if_missing)

        node_table = self.parse_mmcif_ca_residues(
            cif_path=str(cif_path),
            pdb_id=pdb,
            model_index=model_index,
        )
        sifts_ranges = self.fetch_sifts_ranges(pdb)
        node_table = self.attach_uniprot_mapping(node_table, sifts_ranges)

        edge_table = self.build_edge_table(
            node_df=node_table,
            inter_chain_only=inter_chain_only,
            distance_cutoff=distance_cutoff,
        )

        return {
            "node_table": node_table,
            "edge_table": edge_table,
            "sifts_ranges": sifts_ranges,
        }

    def save_quantized_tables(
        self,
        result: Dict[str, pd.DataFrame],
        output_prefix: str,
        to_csv: bool = True,
        to_pickle: bool = True,
        to_parquet: bool = False,
    ) -> Dict[str, str]:
        """把量化结果落盘。"""
        base = Path(output_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {}
        for name, table in result.items():
            if not isinstance(table, pd.DataFrame):
                continue

            if to_csv:
                csv_path = base.with_name(f"{base.name}_{name}.csv")
                self._serialize_df_for_csv(table).to_csv(csv_path, index=False)
                saved[f"{name}_csv"] = str(csv_path)

            if to_pickle:
                pkl_path = base.with_name(f"{base.name}_{name}.pkl")
                table.to_pickle(pkl_path)
                saved[f"{name}_pkl"] = str(pkl_path)

            if to_parquet:
                parquet_path = base.with_name(f"{base.name}_{name}.parquet")
                table.to_parquet(parquet_path, index=False)
                saved[f"{name}_parquet"] = str(parquet_path)

        return saved


if __name__ == "__main__":
    # 示例：默认不限制距离（distance_cutoff=None）
    quantizer = StructureEventQuantizer(root_dir="/data2")
    out = quantizer.quantize_from_pdb_id(
        pdb_id="7W1M",
        inter_chain_only=False,
        distance_cutoff=None,
        download_if_missing=True,
    )

    print("node_table shape:", out["node_table"].shape)
    print("edge_table shape:", out["edge_table"].shape)

    saved_paths = quantizer.save_quantized_tables(
        result=out,
        output_prefix="/data2/7w1m_class_quantized",
        to_csv=True,
        to_pickle=True,
        to_parquet=False,
    )
    print(saved_paths)
