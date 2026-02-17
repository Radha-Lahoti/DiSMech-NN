import numpy as np
import os

GEOMETRY_FLOAT = np.float64
GEOMETRY_INT = np.int64

def from_txt(fname: str):
    """
    File format:
      *nodes
      x,y,z
      ...
      *edges
      i,j
      ...
      *triangles
      i,j,k
      ...

    Returns:
      nodes: (N,3) float64
      edges: (E,2) int64  (0-based)
      faces: (F,3) int64  (0-based)
    """
    if not os.path.isfile(fname):
        raise ValueError(f"{fname} is not a valid file path")

    section = None
    nodes_list, edges_list, faces_list = [], [], []

    with open(fname, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty lines / comments
            if not line or line.startswith("#"):
                continue

            # Header lines
            low = line.lower()
            if low in ("*nodes", "*edges", "*triangles"):
                section = low
                continue

            if section is None:
                raise ValueError(f"Found data before any header: {line}")

            # Skip the column label line in each section
            # (exactly matches your file: x,y,z / i,j / i,j,k)
            if section == "*nodes" and line.replace(" ", "").lower() == "x,y,z":
                continue
            if section == "*edges" and line.replace(" ", "").lower() == "i,j":
                continue
            if section == "*triangles" and line.replace(" ", "").lower() == "i,j,k":
                continue

            vals = [v.strip() for v in line.split(",")]

            if section == "*nodes":
                if len(vals) != 3:
                    raise ValueError(f"Node line must have 3 values, got: {vals}")
                nodes_list.append([float(vals[0]), float(vals[1]), float(vals[2])])

            elif section == "*edges":
                if len(vals) != 2:
                    raise ValueError(f"Edge line must have 2 values, got: {vals}")
                # file is 1-based -> convert to 0-based
                edges_list.append([int(float(vals[0])) - 1, int(float(vals[1])) - 1])

            elif section == "*triangles":
                if len(vals) != 3:
                    raise ValueError(f"Triangle line must have 3 values, got: {vals}")
                # file is 1-based -> convert to 0-based
                faces_list.append([
                    int(float(vals[0])) - 1,
                    int(float(vals[1])) - 1,
                    int(float(vals[2])) - 1
                ])

    nodes = np.asarray(nodes_list, dtype=GEOMETRY_FLOAT).reshape(-1, 3)
    edges = np.asarray(edges_list, dtype=GEOMETRY_INT).reshape(-1, 2)
    faces = np.asarray(faces_list, dtype=GEOMETRY_INT).reshape(-1, 3)

    return nodes, edges, faces
