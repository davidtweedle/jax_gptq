import jax.numpy as jnp

from jax_gptq.pallas.blocked_pivoted_qr import blocked_pivoted_qr


def main() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    work_largest, perm_largest = blocked_pivoted_qr(a, panel_size=2, pivot_mode="largest")
    work_smallest, perm_smallest = blocked_pivoted_qr(a, panel_size=2, pivot_mode="smallest")

    print("largest perm:", perm_largest)
    print("largest work:")
    print(work_largest)
    print()
    print("smallest perm:", perm_smallest)
    print("smallest work:")
    print(work_smallest)


if __name__ == "__main__":
    main()
