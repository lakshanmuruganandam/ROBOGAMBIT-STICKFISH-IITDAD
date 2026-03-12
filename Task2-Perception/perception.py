import math
import cv2
import numpy as np
import sys


class RoboGambit_Perception:

    def __init__(self):
        # PARAMETERS - Camera intrinsics provided by organisers (DO NOT MODIFY)
        self.camera_matrix = np.array([
            [1030.4890823364258, 0, 960],
            [0, 1030.489103794098, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((1, 5))

        # INTERNAL VARIABLES
        self.corner_world = {
            21: (350, 350),
            22: (350, -350),
            23: (-350, -350),
            24: (-350, 350)
        }
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        self.H_matrix = None

        self.board = np.zeros((6, 6), dtype=int)

        self._last_row = 0
        self._last_col = 0

        # ARUCO DETECTOR with tuned parameters for robust detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Tune detector for maximum reliability
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 30
        self.aruco_params.adaptiveThreshWinSizeStep = 5
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.02
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 50
        self.aruco_params.cornerRefinementMinAccuracy = 0.01

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        print("Perception Initialized")


    # DO NOT MODIFY THIS FUNCTION
    def prepare_image(self, image):
        """
        DO NOT MODIFY.
        Performs camera undistortion and grayscale conversion.
        """
        undistorted_image = cv2.undistort(image,self.camera_matrix,self.dist_coeffs,None,self.camera_matrix)
        gray_image = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
        return undistorted_image, gray_image


    def enhance_image(self, gray_image):
        """
        Apply CLAHE contrast enhancement for better marker detection
        in challenging lighting conditions.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        return enhanced


    def detect_markers_robust(self, gray_image):
        """
        Multi-pass marker detection. Tries original, CLAHE-enhanced,
        and adaptive-threshold images. Keeps the pass that detects
        the most markers (preserving duplicate IDs for multiple
        physical pieces sharing the same ArUco ID).
        """
        best_corners, best_ids, best_rejected = None, None, None
        best_count = 0

        # Pass 1: Original grayscale
        c1, i1, r1 = self.detector.detectMarkers(gray_image)
        count1 = 0 if i1 is None else len(i1)
        if count1 > best_count:
            best_corners, best_ids, best_rejected = c1, i1, r1
            best_count = count1

        # Pass 2: CLAHE contrast-enhanced
        enhanced = self.enhance_image(gray_image)
        c2, i2, r2 = self.detector.detectMarkers(enhanced)
        count2 = 0 if i2 is None else len(i2)
        if count2 > best_count:
            best_corners, best_ids, best_rejected = c2, i2, r2
            best_count = count2

        # Pass 3: Adaptive threshold for harsh lighting
        adaptive = cv2.adaptiveThreshold(gray_image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        c3, i3, r3 = self.detector.detectMarkers(adaptive)
        count3 = 0 if i3 is None else len(i3)
        if count3 > best_count:
            best_corners, best_ids, best_rejected = c3, i3, r3
            best_count = count3

        return best_corners, best_ids, best_rejected


    # IMPLEMENTED: PIXEL -> WORLD TRANSFORMATION
    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates into world coordinates using homography.
        """
        if self.H_matrix is None:
            return None, None

        # Format as (1, 1, 2) array for perspectiveTransform
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)

        world_x = float(world_point[0][0][0])
        world_y = float(world_point[0][0][1])

        return world_x, world_y


    def pixel_to_world_batch(self, pixel_points):
        """
        Batch convert multiple pixel coordinates to world coordinates.
        pixel_points: list of (x, y) tuples
        Returns: list of (world_x, world_y) tuples
        """
        if self.H_matrix is None or len(pixel_points) == 0:
            return [(None, None)] * len(pixel_points)

        pts = np.array([[list(p) for p in pixel_points]], dtype=np.float32)
        world_pts = cv2.perspectiveTransform(pts, self.H_matrix)

        results = []
        for i in range(len(pixel_points)):
            wx = float(world_pts[0][i][0])
            wy = float(world_pts[0][i][1])
            results.append((wx, wy))
        return results


    # PARTICIPANTS MODIFY THIS FUNCTION
    def process_image(self, image):
        """
        Main perception pipeline.
        """
        self.board[:] = 0

        # Preprocess image (Do not modify)
        undistorted_image, gray_image = self.prepare_image(image)

        # Step 1: Robust multi-pass ArUco detection with merging
        corners, ids, rejected = self.detect_markers_robust(gray_image)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)

        if ids is None:
            print("No ArUco markers detected.")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        ids_flat = ids.flatten()
        print(f"Detected {len(ids_flat)} markers: {sorted(ids_flat.tolist())}")

        # Step 2: Extract corner marker pixels (IDs 21-24)
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        for i, marker_id in enumerate(ids_flat):
            marker_corners = corners[i][0]  # shape (4, 2)
            cx = float(np.mean(marker_corners[:, 0]))
            cy = float(np.mean(marker_corners[:, 1]))

            if marker_id in self.corner_world:
                self.corner_pixels[marker_id] = (cx, cy)
                self.pixel_matrix.append([cx, cy])
                self.world_matrix.append(list(self.corner_world[marker_id]))

        print(f"Corner markers found: {list(self.corner_pixels.keys())}")

        # Step 3: Compute homography matrix with RANSAC
        if len(self.pixel_matrix) >= 4:
            pixel_pts = np.array(self.pixel_matrix, dtype=np.float32)
            world_pts = np.array(self.world_matrix, dtype=np.float32)
            self.H_matrix, status = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)

            # Validate homography by back-projecting corner markers
            if self.H_matrix is not None:
                reprojection_error = self.validate_homography(pixel_pts, world_pts)
                print(f"Homography computed. Reprojection error: {reprojection_error:.2f} mm")
                if reprojection_error > 50.0:
                    print("WARNING: High reprojection error - results may be inaccurate!")
            else:
                print("ERROR: Homography computation failed!")
        else:
            print(f"WARNING: Only {len(self.pixel_matrix)} corner markers found, need 4.")

        # Step 4: Convert piece markers to world coordinates and place on board
        if self.H_matrix is not None:
            piece_pixels = []
            piece_ids = []

            for i, marker_id in enumerate(ids_flat):
                if 1 <= marker_id <= 10:
                    marker_corners = corners[i][0]
                    cx = float(np.mean(marker_corners[:, 0]))
                    cy = float(np.mean(marker_corners[:, 1]))
                    piece_pixels.append((cx, cy))
                    piece_ids.append(int(marker_id))

            # Batch transform all piece coordinates at once
            if piece_pixels:
                world_coords = self.pixel_to_world_batch(piece_pixels)

                for pid, (wx, wy) in zip(piece_ids, world_coords):
                    if wx is not None:
                        self.place_piece_on_board(pid, wx, wy)

        # Print board state
        print(f"\nReconstructed Board State (6x6):")
        print(self.board)
        print()

        # Save board state to text file
        self.save_board_state()

        # Visualization (Do not modify)
        res = cv2.resize(undistorted_image, (1152, 648))
        cv2.imshow("Detected Markers", res)
        self.visualize_board()


    def validate_homography(self, pixel_pts, world_pts):
        """
        Compute mean reprojection error to validate homography quality.
        """
        pts = pixel_pts.reshape(1, -1, 2)
        projected = cv2.perspectiveTransform(pts, self.H_matrix)
        projected = projected.reshape(-1, 2)
        expected = world_pts.reshape(-1, 2)
        errors = np.linalg.norm(projected - expected, axis=1)
        return float(np.mean(errors))


    # IMPLEMENTED: BOARD PLACEMENT
    def place_piece_on_board(self, piece_id, x_coord, y_coord):
        """
        Places detected piece on the closest board square using
        minimum Euclidean distance matching to cell centers.

        Board definition:
          6x6 grid, top-left corner = (300, 300), square size = 100mm

        Cell centers in world coordinates:
          x: 250, 150, 50, -50, -150, -250  (col 0 -> 5)
          y: 250, 150, 50, -50, -150, -250  (row 0 -> 5)
        """
        cell_centers = [250, 150, 50, -50, -150, -250]

        # Find nearest cell by Euclidean distance to cell center
        best_row = 0
        best_col = 0
        min_dist = float('inf')

        for r in range(6):
            for c in range(6):
                dx = x_coord - cell_centers[c]
                dy = y_coord - cell_centers[r]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
                    best_row = r
                    best_col = c

        self._last_row = best_row
        self._last_col = best_col

        # Only place if within reasonable Euclidean distance from cell center
        if min_dist < 60:
            # Check for collision - warn if overwriting
            if self.board[best_row][best_col] != 0:
                print(f"  WARNING: Cell [{best_row},{best_col}] already occupied by piece "
                      f"{self.board[best_row][best_col]}, overwriting with {piece_id}")
            self.board[best_row][best_col] = piece_id
            print(f"  Piece {piece_id:2d} -> world ({x_coord:7.1f}, {y_coord:7.1f}) "
                  f"-> board [{best_row},{best_col}] (dist: {min_dist:.1f}mm)")
        else:
            print(f"  WARNING: Piece {piece_id} at ({x_coord:.1f}, {y_coord:.1f}) "
                  f"too far from nearest cell center ({min_dist:.1f}mm) - skipped")


    def save_board_state(self):
        """Save board state to a text file for submission."""
        piece_names = {
            0: "Empty", 1: "White Pawn", 2: "White Knight", 3: "White Bishop",
            4: "White Queen", 5: "White King", 6: "Black Pawn", 7: "Black Knight",
            8: "Black Bishop", 9: "Black Queen", 10: "Black King"
        }
        files = "ABCDEF"

        with open("board_output.txt", "w") as f:
            f.write("=" * 50 + "\n")
            f.write("RoboGambit - Board State Estimation Output\n")
            f.write("=" * 50 + "\n\n")

            f.write("Board Array (6x6 NumPy):\n")
            f.write(str(self.board) + "\n\n")

            f.write("Piece Positions:\n")
            f.write("-" * 40 + "\n")
            for r in range(6):
                for c in range(6):
                    piece = int(self.board[r][c])
                    if piece != 0:
                        rank = r + 1
                        file_letter = files[c]
                        cell = f"{file_letter}{rank}"
                        f.write(f"  {cell}: {piece_names.get(piece, f'ID {piece}')} (ID={piece})\n")

            f.write("\nVisual Board:\n")
            f.write("   A  B  C  D  E  F\n")
            for r in range(5, -1, -1):
                rank = r + 1
                row_str = f"{rank}  "
                for c in range(6):
                    piece = int(self.board[r][c])
                    if piece == 0:
                        row_str += " . "
                    else:
                        row_str += f"{piece:2d} "
                f.write(row_str + "\n")

        print("Board state saved to board_output.txt")


    # DO NOT MODIFY THIS FUNCTION
    def visualize_board(self):
        """
        Draw a simple 6x6 board with detected piece IDs
        """
        cell_size = 80
        board_img = np.ones((6*cell_size,6*cell_size,3),dtype=np.uint8) * 255

        for r in range(6):
            for c in range(6):
                x1 = c*cell_size
                y1 = r*cell_size
                x2 = x1+cell_size
                y2 = y1+cell_size
                cv2.rectangle(board_img,(x1,y1),(x2,y2),(0,0,0),2)

                piece = int(self.board[r][c])
                if piece != 0:
                    cv2.putText(board_img,str(piece),(x1+25,y1+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Game Board", board_img)


# DO NOT MODIFY
def main():
    # To run code, use python/python3 perception.py path/to/image.png
    if len(sys.argv) < 2:
        print("Usage: python perception.py image.png")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    perception = RoboGambit_Perception()
    perception.process_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
