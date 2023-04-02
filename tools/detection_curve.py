import matplotlib.pyplot as plt

# test data
red_baseline = [None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, 0.3,
                None, None, None, None, None]
res34_retinanet_baseline = [0.0795, 0.192, 0.226, 0.179, 0.203, 0.231, 0.23, 0.224, 0.262, 0.244,
                            0.253, 0.264, 0.248, 0.244, 0.268, 0.264, 0.257, 0.253, 0.259, 0.253,
                            None, None, None, None, None]
res18_retinanet_baseline = [0.087, 0.162, 0.148, 0.213, 0.224, 0.209, 0.228, 0.238, 0.226, 0.223,
                            0.248, 0.22, 0.225, 0.217, 0.256, 0.256, 0.257, 0.243, 0.245, 0.244,
                            None, None, None, None, None, None, None, None, None, None]
recurrent_res18_retinanet_v1 = [0.07, 0.0856, 0.152, 0.188, 0.163, 0.23, 0.21, 0.172, 0.176, 0.224,
                             0.233, 0.236, 0.234, 0.241, 0.234, 0.229, 0.225, 0.242, 0.241, 0.221,
                             0.254, 0.22, 0.238, 0.237, None]
recurrent_res18_retinanet_v2 = [0.088, 0.12, 0.142, 0.188, 0.202, 0.207, 0.19, 0.203, 0.208, 0.229,
                                0.243, 0.239, 0.229, 0.216, 0.23, 0.259, 0.249, 0.23, 0.245, 0.254,
                                0.239, 0.242, 0.243, 0.221, 0.243]
recurrent_res18_retinanet_v3 = [0.08, 0.144, 0.174, 0.171, 0.174, 0.23, 0.212, 0.195, 0.236, 0.245,
                                0.22, 0.258, 0.252, 0.245, 0.253, 0.228, 0.258, 0.28, 0.258, 0.271,
                                0.241, 0.245, 0.275, 0.266, 0.262, 0.266, 0.27, 0.262, None, None]
up_lstm_v1 = [0.068, 0.109, 0.11, 0.156, 0.187, 0.208, 0.164, 0.215, 0.177, 0.211,
              0.224, 0.214, 0.23, 0.249, 0.222, 0.239, 0.254, 0.215, 0.25, 0.248,
              0.255, 0.249, 0.241, 0.214, 0.276, 0.255, 0.244, 0.259, 0.259, 0.256]
up_lstm_v2 = [0.043, 0.107, 0.123, 0.17, 0.188, 0.202, 0.199, 0.209, 0.195, 0.23,
              0.221, 0.207, 0.224, 0.246, 0.231, 0.252, 0.235, 0.234, 0.233, 0.229,
              0.24, 0.27, 0.244, 0.23, 0.23, 0.24, 0.233, 0.233, 0.241, 0.242]
recurrent_res18_retinanet_v3_hidden_fusion = [0, 0.018, 0.055, 0.0685, 0.078, 0.136, 0.15, 0.209, 0.167, 0.222,
                                              0.23, 0.153, 0.228, 0.151, 0.191, 0.251, 0.251, 0.268, 0.256, 0.269,
                                              0.251, 0.279, 0.208, 0.273, 0.286, 0.3, 0.277, 0.261, 0.288, 0.257]


# x-axis
axis_x = [i+1 for i in range(len(res18_retinanet_baseline))]  # 30 epoch


# create figure
fig = plt.figure()
# draw figure
# plt.plot(axis_x, res34_retinanet_baseline, '-r', linewidth=1, label="res34-retinanet")
plt.plot(axis_x, res18_retinanet_baseline, '-y', linewidth=1, label="res18-retinanet")
# plt.plot(axis_x, up_lstm_v1, '-b', linewidth=1, label="up_lstm_v1")
# plt.plot(axis_x, up_lstm_v2, '-p', linewidth=1, label="up_lstm_v2")
# plt.plot(axis_x, red_baseline, '-p', linewidth=1, label="red baseline", marker="o")
plt.plot(axis_x, recurrent_res18_retinanet_v3, '-b', linewidth=1, label="recurrent_res18")
plt.plot(axis_x, recurrent_res18_retinanet_v3_hidden_fusion, '-r', linewidth=1, label="recurrent_res18_hidden_fusion")

# title
plt.title("mAP@0.5 curve")

plt.xlabel("epoch")
plt.ylabel("mAP@0.5")
plt.legend(loc="lower right")

plt.show()
