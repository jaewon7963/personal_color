import pandas as pd

df_swb = pd.read_csv('./csv/swb.csv')
df_swl = pd.read_csv('./csv/swl.csv')
df_scm = pd.read_csv('./csv/scm.csv')
df_scl = pd.read_csv('./csv/scl.csv')
df_fwd = pd.read_csv('./csv/fwd.csv')
df_fwm = pd.read_csv('./csv/fwm.csv')
df_wcd = pd.read_csv('./csv/wcd.csv')
df_wcb = pd.read_csv('./csv/wcb.csv')

df = pd.concat([df_swb, df_swl, df_scm, df_scl, df_fwd, df_fwm, df_wcd, df_wcb], ignore_index = True)

df.to_csv('color_cloth.csv', index=False)