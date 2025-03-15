import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1
array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

array2 = array1 + 10

array3 = array1 * np.array([[0.5, 1, 1.5], [2, 2.5, 3], [3.5, 4, 4.5]])


print("Array 2:")
print("dtype:", array2.dtype)
print("shape:", array2.shape)
print("size:", array2.size)

print("\nArray 3:")
print("dtype:", array3.dtype)
print("shape:", array3.shape)
print("size:", array3.size)

# 2
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 6, 7, 8, np.nan],
    'C': [10, 11, 12, np.nan, 14],
    'D': [15, np.nan, 17, 18, 19]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

print("\nMissing values (isna):\n", df.isna())
print("\nMissing values (isnull):\n", df.isnull())

df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:\n", df_dropped)

df_filled_zero = df.fillna(0)
print("\nDataFrame filled with 0:\n", df_filled_zero)

df_filled_mean = df.fillna(df.mean())
print("\nDataFrame filled with mean:\n", df_filled_mean)

# 3
time = np.arange(0, 10, 0.5)  # Time points from 0 to 9 with step 0.5
values = np.sin(time) * time  # Example data: sine wave multiplied by time

plt.figure(figsize=(10, 5)) # Set figure size
plt.plot(time, values)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Variable Change Over Time")

plt.plot(time, values, color='red', marker='o', linestyle='--', linewidth=2)

age = np.random.randint(18, 65, 50)  # Generate random ages between 18 and 64
income = age * 1000 + np.random.normal(0, 10000, 50)  # Generate income related to age with some noise

plt.figure(figsize=(10, 5)) # Set figure size for scatter plot
plt.scatter(age, income, color='blue', marker='x')
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs. Income")

plt.show()