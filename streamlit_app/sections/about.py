from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Adaptive Optimization Algorithms from Scratch")
    st.caption(
        "First-order and quasi-Newton optimization algorithms implemented from scratch "
        "in NumPy -- no PyTorch, no TensorFlow, no autograd."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimizers implemented", "7")
    col2.metric("Benchmark functions", "4")
    col3.metric("Datasets", "a9a · MNIST · synthetic")

    st.markdown(
        """
This dashboard is the interactive companion to the **Adaptive Optimization
Algorithms from Scratch** library. Every optimizer -- Gradient Descent, SGD,
Adagrad, AdaGrad-Norm, RMSProp, Adam, and L-BFGS -- is implemented in pure
NumPy with a shared, minimal interface:

```python
class Optimizer(ABC):
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        ...
```

Use the sidebar to explore:

- **Benchmark Playground** -- watch optimizers race across the Rosenbrock,
  Himmelblau, Beale and Rastrigin functions with contour plots and animated
  trajectories.
- **Train a Classifier** -- fit logistic/softmax regression on a9a, MNIST,
  or a synthetic dataset with any optimizer, live loss/accuracy curves, and
  a confusion matrix.
- **Compare Optimizers** -- train several optimizers side-by-side on the
  same task and get an auto-generated comparison table (loss, accuracy,
  gradient norm, runtime, memory, convergence speed, stability).
"""
    )

    st.subheader("Update equations")
    eq_cols = st.columns(2)
    with eq_cols[0]:
        st.markdown("**Gradient Descent**")
        st.latex(r"w_{t+1} = w_t - \eta \nabla f(w_t)")
        st.markdown("**SGD (+ momentum)**")
        st.latex(r"v_{t+1} = \mu v_t + g_t, \quad w_{t+1} = w_t - \eta v_{t+1}")
        st.markdown("**Adagrad**")
        st.latex(r"G_t = G_{t-1} + g_t^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t}+\varepsilon} g_t")
        st.markdown("**AdaGrad-Norm**")
        st.latex(
            r"b_t^2 = b_{t-1}^2 + \lVert g_t \rVert^2, \quad "
            r"w_{t+1} = w_t - \frac{\eta}{b_t+\varepsilon} g_t"
        )
    with eq_cols[1]:
        st.markdown("**RMSProp**")
        st.latex(
            r"v_t = \rho v_{t-1} + (1-\rho) g_t^2, \quad "
            r"w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t}+\varepsilon} g_t"
        )
        st.markdown("**Adam**")
        st.latex(
            r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad "
            r"v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2"
        )
        st.latex(
            r"w_{t+1} = w_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}, \quad "
            r"\hat m_t = \frac{m_t}{1-\beta_1^t}, \ \hat v_t = \frac{v_t}{1-\beta_2^t}"
        )
        st.markdown("**L-BFGS**")
        st.latex(r"d_t = -H_t \nabla f(w_t) \ \text{(two-loop recursion)}, \quad w_{t+1} = w_t + \eta_t d_t")

    st.info(
        "See the project **README** for the full mathematical background, "
        "advantages/disadvantages of each method, and experimental results "
        "on a9a and MNIST.",
        icon="📖",
    )
