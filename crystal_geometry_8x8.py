import numpy as np
import os

class CrystalGeometryTemplate:
    """
    Handles the fixed crystal geometry template and insertion of a 4x4x6
    sub-block (expanded to 8x8x6) into that template, with lead shielding
    strictly attached outside the module.

    The pipeline matches the logic:
    - Template: 35 x 35 x 17 (map_new2)
    - Bottom crystal layer: 8 x 8 x 2 at z=0:2
    - Dynamic region: 8 x 8 x 6 block at z=11:17
    - Lead shielding: 1-voxel boundary directly adjacent to 8x8 module
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
        Build the static 35x35x17 geometry template with an 8x8 module:
        - 8x8x2 bottom crystal (material 2)
        - 8x8x6 dynamic region marked by SENTINEL (-1)
        - 1-voxel lead (material 3) boundary strictly attached on all sides
        """
        S = self.SENTINEL

        # 10x10 = 8x8 inner + 1-voxel lead boundary each side
        map_new = np.zeros((10, 10, 17), dtype=np.int8)
        map_new[1:9, 1:9, 4:6] = 2  # bottom crystal 8x8x2

        # Dynamic 8x8x6 region (marked SENTINEL for agent optimisation)
        map_new[1:9, 1:9, 11:17] = S

        # Lead boundaries — strictly attached to the 8x8 module
        map_new[:, 0, :] = 3
        map_new[:, 9, :] = 3
        map_new[0, :, :] = 3
        map_new[9, :, :] = 3

        # Remaining 0s become 4 (acrylic)
        zero_mask = (map_new == 0)
        map_new[zero_mask] = 4

        # Outer lead frame
        map_new_tmp = np.full((12, 12, 17), 3, dtype=np.int8)
        map_new_tmp[1:11, 1:11, :] = map_new

        # Hollow interior
        map_new_tmp[4:12, 4:12, 6:11] = 0

        map_new2 = np.zeros((35, 35, 17), dtype=np.int8)
        map_new2[12:24, 12:24, :] = map_new_tmp

        return map_new2

    def _find_dynamic_region_slices(self):
        """
        Detect the bounding box of the SENTINEL region in the template,
        and return it as (xslice, yslice, zslice).
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
        Take a 4x4x6 sub-block and place it into 4 quadrants of an 8x8x6
        block with 90-degree clockwise rotations for radial symmetry.

        Quadrant layout (viewed from above):
          TL (0°)   | TR (90° CW)
          ----------+------------
          BL (270°) | BR (180°)
        """
        sub_block = np.asarray(sub_block, dtype=np.int8).reshape((4, 4, 6), order="F")

        block = np.zeros((8, 8, 6), dtype=np.int8)

        # Top-left: original
        block[0:4, 0:4, :] = sub_block
        # Top-right: 90° clockwise (k=-1 in np.rot90)
        block[0:4, 4:8, :] = np.rot90(sub_block, k=-1, axes=(0, 1))
        # Bottom-right: 180°
        block[4:8, 4:8, :] = np.rot90(sub_block, k=-2, axes=(0, 1))
        # Bottom-left: 270° clockwise
        block[4:8, 0:4, :] = np.rot90(sub_block, k=-3, axes=(0, 1))

        return block

    def apply_module(self, sub_block: np.ndarray):
        """
        Insert a given 4x4x6 pattern (tiled to 8x8x6) into the template and return:
          - full_map: 35x35x17 array
          - cube_pos: (N,3) array of [x,y,z] coordinates where full_map==1 or 2
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

    sub_block = np.zeros((4, 4, 6), dtype=np.int8)
    sub_block[0, 0, 0] = 1
    sub_block[1, 1, 1] = 1
    sub_block[2, 2, 2] = 1
    sub_block[3, 3, 3] = 1
    sub_block[0, 1, 4] = 1
    sub_block[1, 0, 5] = 1

    full_map, cube_pos = geom.apply_module(sub_block)

    nz = full_map.shape[2]
    assert full_map.shape == (35, 35, 17)
    map_reshape = np.zeros((35 * nz, 35), dtype=full_map.dtype)
    for layer in range(nz):
        start = layer * 35
        end = (layer + 1) * 35
        map_reshape[start:end, :] = full_map[:, :, layer]

    os.makedirs("debug", exist_ok=True)
    cube_pos_path = "debug/cube_pos_test_8x8.txt"
    np.savetxt(cube_pos_path, cube_pos, fmt="%5.2f")

    map_path = "debug/map_test_8x8.txt"
    np.savetxt(map_path, map_reshape, fmt="%2d")
    print(f"Template shape: {full_map.shape}")
    print(f"Crystal voxels: {cube_pos.shape[0]}")
    print(f"Dynamic region slices: {geom._dyn_slices}")
