import numpy as np
import os

class CrystalGeometryTemplate:
    """
    Handles the fixed crystal geometry template and insertion of a 3x3x3
    module (expanded to 9x9x6) into that template.

    The pipeline matches the logic:
    - Template: 35 x 35 x 17 (map_new2)
    - Dynamic region: 9 x 9 x 6 block embedded near the "top" (z-large)
    """

    SENTINEL = -1 

    def __init__(self, dimx=3, dimy=3, dimz=3,
                 x_range=(-51, 51),
                 y_range=(-51, 51),
                 z_range=(-42, 6)):
        # Coordinate grids 
        self.x_new = np.arange(x_range[0], x_range[1] + dimx, dimx, dtype=np.float32)
        self.y_new = np.arange(y_range[0], y_range[1] + dimy, dimy, dtype=np.float32)
        self.z_new = np.arange(z_range[0], z_range[1] + dimz, dimz, dtype=np.float32)

        assert self.x_new.size == 35
        assert self.y_new.size == 35
        assert self.z_new.size == 17

        # Build template once and cache dynamic region slice
        self.template = self._build_template()
        self._dyn_slices = self._find_dynamic_region_slices()

    def _build_template(self) -> np.ndarray:
        """
        Build the static 35x35x17 geometry template, with the 9x9x6 region marked by SENTINEL (-1). 
        """
        S = self.SENTINEL

        map_new = np.zeros((11, 11, 17), dtype=np.int8)
        map_new[1:10, 1:10, 9:11] = 2

        # Dynamic 9x9x6 region (we mark with SENTINEL for now)
        map_new[1:10, 1:10, 11:17] = S

        # Boundaries 
        map_new[:, 0, :] = 3
        map_new[:, 10, :] = 3
        map_new[0, :, :] = 3
        map_new[10, :, :] = 3

        # Remaining 0s become 4 (acrylic)
        zero_mask = (map_new == 0)
        map_new[zero_mask] = 4

        map_new_tmp = np.full((13, 13, 17), 3, dtype=np.int8)
        map_new_tmp[1:12, 1:12, :] = map_new

        # Hollow interior
        map_new_tmp[0:9, 0:9, 0:9] = 0

        map_new2 = np.zeros((35, 35, 17), dtype=np.int8)
        map_new2[11:24, 11:24, :] = map_new_tmp

        return map_new2

    def _find_dynamic_region_slices(self):
        """
        Detect the bounding box of the SENTINEL region in the template, and return it as (xslice, yslice, zslice). This is where the 9x9x6
        block will be written.
        """
        coords = np.where(self.template == self.SENTINEL)

        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        z_min, z_max = coords[2].min(), coords[2].max()

        xs = slice(x_min, x_max + 1)
        ys = slice(y_min, y_max + 1)
        zs = slice(z_min, z_max + 1)

        return xs, ys, zs

    def _build_full_block(self, sub_block: np.ndarray) -> np.ndarray:
        """
        Take a 3x3x6 array and expand it to a 9x9x6 array 
        """
        sub_block = np.asarray(sub_block, dtype=np.int8).reshape((3, 3, 6), order="F")

        base = sub_block.reshape(-1, order="F")         
        num_sub_block = 3*3*1                     
        full_map = np.tile(base[:, None], (1, num_sub_block)) 

        tmp = full_map.reshape((3, 3, 6, 3, 3, 1), order="F")
        tmp = np.transpose(tmp, (0, 3, 1, 4, 2, 5))  
        block = tmp.reshape((9, 9, 6), order="F")

        return block

    def apply_module(self, sub_block: np.ndarray):
        """
        Insert a given 3x3x3 pattern into the template and return:
          - full_map: 35x35x17 array
          - cube_pos: (N,3) array of [x,y,z] coordinates where full_map==1 or 2(if return_cube_pos=True)
        """
        block = self._build_full_block(sub_block)
        full_map = self.template.copy()

        xs, ys, zs = self._dyn_slices
        full_map[xs, ys, zs] = block

        # Compute crystal coordinates: map==1 or map==2
        aa, bb, cc = np.where((full_map == 1) | (full_map == 2))
        cube_pos = np.column_stack([
            self.x_new[aa],
            self.y_new[bb],
            self.z_new[cc],
        ]).astype(np.float32)

        return full_map, cube_pos

    def write_geometry(self, full_map: np.ndarray, cube_pos: np.ndarray, out_dir: str, idx: int) -> None:
        """
        Save full_map and cube_pos to text files
        """
        os.makedirs(out_dir, exist_ok=True)
        assert full_map.shape == (35, 35, 17), f"Unexpected full_map shape: {full_map.shape}"
        nz = full_map.shape[2]

        map_reshape = np.zeros((35 * nz, 35), dtype=full_map.dtype)
        for layer in range(nz):
            start = layer * 35
            end = (layer + 1) * 35
            map_reshape[start:end, :] = full_map[:, :, layer]

        cube_pos_path = os.path.join(out_dir, f"cube_pos_{idx:03d}.txt")
        np.savetxt(cube_pos_path, cube_pos, fmt="%5.2f")

        map_path = os.path.join(out_dir, f"map_{idx:03d}.txt")
        np.savetxt(map_path, map_reshape, fmt="%2d")


if __name__ == "__main__":
    geom = CrystalGeometryTemplate()

    sub_block = np.zeros((3, 3, 6), dtype=np.int8)
    sub_block[0, 0, 0] = 1
    sub_block[1, 1, 1] = 1
    sub_block[2, 2, 2] = 1
    sub_block[0, 0, 3] = 1
    sub_block[1, 1, 4] = 1
    sub_block[2, 2, 5] = 1

    full_map, cube_pos = geom.apply_module(sub_block)

    nz = full_map.shape[2]
    assert full_map.shape == (35, 35, 17)
    map_reshape = np.zeros((35 * nz, 35), dtype=full_map.dtype)
    for layer in range(nz):
        start = layer * 35
        end = (layer + 1) * 35
        map_reshape[start:end, :] = full_map[:, :, layer]
    
    cube_pos_path = "cube_test.txt"
    np.savetxt(cube_pos_path, cube_pos, fmt="%5.2f")

    map_path = "map_test.txt"
    np.savetxt(map_path, map_reshape, fmt="%2d")


