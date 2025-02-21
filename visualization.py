import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import sklearn.preprocessing as PrePro

Train = 0
Dataset = 'MAIN_Anno_sort_h'

if Train:
    with open(f"./Feature_Train/Feature_{Dataset}.pkl", "rb") as f:
        feature_vector = pickle.load(f)

    # Scale and reduce dimensions with PCA
    scaler = PrePro.MinMaxScaler()
    pca = PCA(n_components=2)
    clf = OneClassSVM(kernel='rbf', gamma=0.0005, nu=0.1)

    s = np.concatenate(feature_vector, axis=0).shape
    feature_vector = np.concatenate(feature_vector, axis=0).reshape(-1, s[1]*s[2]*s[3]*s[4])

    scaler.fit(feature_vector)
    feature_vector = scaler.transform(feature_vector)
    pca.fit(feature_vector)
    pca_feature = pca.transform(feature_vector)
    clf.fit(pca_feature)
    pred = clf.predict(pca_feature)
    print(f"pred = 1 : {np.sum(pred == 1)}, pred = -1 : {np.sum(pred == -1)}")

    distance_threshold = 100  # 이 임계값을 조정해 포인트 필터링
    distance_from_origin = np.sqrt(np.sum(pca_feature ** 2, axis=1))
    filtered_indices = distance_from_origin < distance_threshold
    filtered_pca_feature = pca_feature[filtered_indices]
    filtered_pred = pred[filtered_indices]

    # x_min, x_max = (filtered_pca_feature[:, 0].min())*1.5 - 1, (filtered_pca_feature[:, 0].max())*1.5 + 1
    # y_min, y_max = (filtered_pca_feature[:, 1].min())*1.5 - 1, (filtered_pca_feature[:, 1].max())*1.5 + 1
    x_min, x_max = -150, 150
    y_min, y_max = -150, 150

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    # Plot decision boundary and points
    plt.figure(figsize=(10, 10))
    levels = np.linspace(Z.min(), 0, 7) if Z.min() < 0 else np.linspace(0, Z.max(), 7)
    plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    colors = np.where(filtered_pred == 1, 'blue', 'red')

    plt.scatter(filtered_pca_feature[:, 0], filtered_pca_feature[:, 1], c=colors, label='Data Points')
    plt.title('OneClass SVM Decision Boundary and Filtered Data Points (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Plot/Plot.png')
    with open(f"./OCS.pkl", "wb") as f:
        pickle.dump({'scaler': scaler, 'pca': pca, 'clf': clf}, f, protocol=pickle.HIGHEST_PROTOCOL)

# Test
else:
    import utils
    data_root = f'./dataset/{Dataset}_T/'
    video_list = utils.get_file_list(data_root)
    video_num = len(video_list)
    ##
    for ite_vid in range(video_num):
        video_name = video_list[ite_vid]
        print(video_name)
        with open(f"./Feature/Feature_{Dataset}/{video_name}", "rb") as f: #
            feature_vector = pickle.load(f)
        with open(f"./OCS.pkl", "rb") as f:
            ocs = pickle.load(f)

        clf = ocs['clf']
        scaler = ocs['scaler']
        pca = ocs['pca']
        fv = feature_vector[0].size
        feature = np.concatenate(feature_vector, axis=0).reshape(-1, fv)
        feature = scaler.transform(feature)
        pca_feature = pca.transform(feature)
        pred = clf.predict(pca_feature)

        print(f"pred = 1 : {np.sum(pred == 1)}, pred = -1 : {np.sum(pred == -1)}")
        negative_pred_indices = np.where(pred == -1)[0]
        print(negative_pred_indices)
        # with open(f"{data_root}{video_name}", "rb") as f: #
        #     data = pickle.load(f)
        # for idx in negative_pred_indices:
        #     selected_data = data[idx]
        #     fig, axes = plt.subplots(1, 16, figsize=(16, 8))  # Adjust the figsize if necessary
        #     for j in range(16):  # Assuming there are 16 slices
        #         mid_slice = selected_data[:, :, j]
        #         axes[j].imshow(mid_slice, cmap='gray', aspect='equal')
        #         axes[j].set_title(f'Slice {j + 1}')
        #         axes[j].axis('off')  # Turn off axis labels
        #
        #     plt.tight_layout()
        #     plt.savefig(f"./Plot/{video_name[:-4]}_{idx}.png")

        distance_threshold = 100  # 이 임계값을 조정해 포인트 필터링
        distance_from_origin = np.sqrt(np.sum(pca_feature ** 2, axis=1))
        filtered_indices = distance_from_origin < distance_threshold
        filtered_pca_feature = pca_feature[filtered_indices]
        filtered_pred = pred[filtered_indices]

        # x_min, x_max = (filtered_pca_feature[:, 0].min()) * 2.5 - 1, (filtered_pca_feature[:, 0].max()) * 2.5 + 1
        # y_min, y_max = (filtered_pca_feature[:, 1].min()) * 2.5 - 1, (filtered_pca_feature[:, 1].max()) * 2.5 + 1
        x_min, x_max = -150, 150
        y_min, y_max = -150, 150

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.decision_function(grid_points)
        Z = Z.reshape(xx.shape)


        levels = np.linspace(Z.min(), Z.max(), 7)
        levels = np.sort(levels)  # Ensure levels are sorted in increasing order
        plt.figure(figsize=(10, 10))
        plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.PuBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

        # Scatter plot for data points
        colors = np.where(filtered_pred == 1, 'blue', 'red')
        plt.scatter(filtered_pca_feature[:, 0], filtered_pca_feature[:, 1], c=colors, label='Data Points')
        plt.title('OneClass SVM Decision Boundary and Training Data Points (PCA)')
        plt.xlabel(f'{video_name[:-4]}')
        # plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./Plot/Plot_{video_name[:-4]}.png')

