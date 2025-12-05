#%%
import numpy as np
npy_path = "/home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/tmp/eval_results_known_skeleton/real/hand_hand/testing/user_05/recording_00.npy"

data = np.load(npy_path, allow_pickle=True)
#%%
print(data.keys())
# %%
print(data['tracked_keypoints'].shape)
print(data['valid_tracking'].shape)
# %%
print(data['tracked_keypoints'][0, 0])
# %%
print(data['valid_tracking'][0, 0])
# %%
print(data['gt_keypoints'][0, 0])
# %%
print(data['gt_keypoints'].shape)
# %%