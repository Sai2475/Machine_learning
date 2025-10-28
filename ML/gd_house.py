import pandas as pd
import numpy as np

def gradient_descent(x, y, lr=0.01, epochs=3000):
    # Min-Max Scaling
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    m, b = 0.0, 0.0

    for epoch in range(epochs):
        y_pred = m * x_scaled + b
        error = y_scaled - y_pred
        cost = np.mean(error**2)

        dm = -2 * np.mean(error * x_scaled)
        db = -2 * np.mean(error)

        m -= lr * dm
        b -= lr * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.6f}, b = {b:.4f}, m = {m:.4f}")

    # Scale back to original space
    m_original = m * (y_max - y_min) / (x_max - x_min)
    b_original = (y_max - y_min) * (b - m * x_min / (x_max - x_min)) + y_min

    return b_original, m_original


if __name__ == '__main__':
    df = pd.read_csv("home_prices.csv")
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    b, m = gradient_descent(x, y)

    print(f"Final Results: y = {m:.2f}x + {b:.2f}")
