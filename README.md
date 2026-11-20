# python
"""
Linear Algebra Operations & Vector Calculations Toolkit
Author: Raghad Al-Jrees
Description: Core linear algebra algorithms and vector space operations 
             implemented using Python and NumPy for foundational machine learning.
"""

import numpy as np


class VectorOperations:
    """A toolkit for core vector calculations and projections."""

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes the algebraic dot product of two vectors."""
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same dimensions.")
        return float(np.dot(v1, v2))

    @staticmethod
    def euclidean_norm(v: np.ndarray) -> float:
        """Computes the L2 norm (magnitude) of a vector."""
        return float(np.linalg.norm(v))

    @staticmethod
    def vector_projection(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Projects vector v onto vector u.
        Formula: proj_u(v) = (v . u / ||u||^2) * u
        """
        u_norm_sq = np.dot(u, u)
        if np.isclose(u_norm_sq, 0.0):
            raise ValueError("Cannot project onto a zero vector.")
        scalar_factor = np.dot(v, u) / u_norm_sq
        return scalar_factor * u

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the cosine of the angle between two vectors."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if np.isclose(norm_v1, 0.0) or np.isclose(norm_v2, 0.0):
            raise ValueError("Cosine similarity undefined for zero vectors.")
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))


class MatrixAnalysis:
    """Linear transformations and matrix column space exploration."""

    @staticmethod
    def column_space_basis(matrix: np.ndarray) -> np.ndarray:
        """
        Extracts the orthonormal basis for the column space of a matrix using SVD.
        """
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = np.sum(s > 1e-10)
        return u[:, :rank]


if __name__ == "__main__":
    # Test vectors
    vec_a = np.array([3.0, 4.0, 0.0])
    vec_b = np.array([1.0, 2.0, 2.0])

    print("=== Linear Algebra Toolkit Demo ===")
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Dot Product: {VectorOperations.dot_product(vec_a, vec_b)}")
    print(f"Norm of A: {VectorOperations.euclidean_norm(vec_a)}")
    print(f"Projection of A onto B: {VectorOperations.vector_projection(vec_a, vec_b)}")
    print(f"Cosine Similarity: {VectorOperations.cosine_similarity(vec_a, vec_b):.4f}")

    # Matrix demo
    sample_mat = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    basis = MatrixAnalysis.column_space_basis(sample_mat)
    print("\nColumn Space Basis:")
    print(basis)
