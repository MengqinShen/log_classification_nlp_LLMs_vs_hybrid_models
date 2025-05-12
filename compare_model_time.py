# import subprocess
# import time
#
# scripts = ["classify_ft_BERT.py", "classify_hybrid.py"]
# input_file = "resources/test.csv"
#
# for script in scripts:
#     start = time.time()
#     subprocess.run(["python", script, input_file], check=True)
#     end = time.time()
#     print(f"{script} ran in {end - start:.4f} seconds")
import pandas as pd

df1 = pd.read_csv("resources/output_ft.csv")
df2 = pd.read_csv("resources/output.csv")

common = pd.merge(df1, df2, how='inner')
diff = df1.compare(df2)
print(common, diff)
