image = imread('parrots.jpg')
float_image = img_as_float(image)
n = 713 * 474
p = np.array(float_image)
s = p.reshape(n, 3)
pass
for N_CLUSTERS in range(1, 20):
    kmeans = KMeans(random_state=241, init='k-means++', n_clusters=N_CLUSTERS)
    kmeans.fit(s)
    new_image = s.copy()
    print(kmeans.n_clusters)
    clusters = []
    for i in range(kmeans.n_clusters):
        clusters.append([])

    i = 0
    for pixel in kmeans.labels_:
        clusters[pixel].append(i)
        i += 1
    count = 0
    for cluster in clusters:
        avgR = 0
        avgG = 0
        avgB = 0
        for pixel in cluster:
            avgR += s[pixel][0]
            avgG += s[pixel][1]
            avgB += s[pixel][2]
            count += 1
        avgR /= len(cluster)
        avgG /= len(cluster)
        avgB /= len(cluster)
        for pixel in cluster:
            new_image[pixel] = [avgR, avgG, avgB]

    MSE = 0
    for i in range(n):
        for color in range(3):
            MSE += (new_image[i][color] - s[i][color]) ** 2

    MSE /= (3 * n)

    PSNR = 10 * np.log10(1 / MSE)
    print(PSNR)
    if PSNR > 20:
        print("FOUND!! ", N_CLUSTERS)
    MSE *= 3
    PSNR = 10 * np.log10(1 / MSE)
    print(PSNR)
    if PSNR > 20:
        print("FOUND!! ", N_CLUSTERS)
