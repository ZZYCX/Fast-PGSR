import open3d as o3d

import numpy as np

import sys





def analyze_mesh(path):



    print("Loading:", path)



    mesh = o3d.io.read_triangle_mesh(path)



    if not mesh.has_triangles():

        print("Mesh has no triangles")

        return



    print("\n===== Original Mesh =====")

    print("Vertices:", len(mesh.vertices))

    print("Triangles:", len(mesh.triangles))





    # connected components

    triangle_clusters, cluster_n_triangles, cluster_area = (

        mesh.cluster_connected_triangles()

    )



    triangle_clusters = np.asarray(triangle_clusters)

    cluster_n_triangles = np.asarray(cluster_n_triangles)

    cluster_area = np.asarray(cluster_area)





    print("\n===== Components =====")

    print("Number of components:", len(cluster_n_triangles))





    vertices = np.asarray(mesh.vertices)

    triangles = np.asarray(mesh.triangles)





    for i in range(len(cluster_n_triangles)):



        tri_idx = np.where(triangle_clusters == i)[0]



        if len(tri_idx) == 0:

            continue



        verts_idx = np.unique(

            triangles[tri_idx].reshape(-1)

        )



        pts = vertices[verts_idx]



        min_xyz = pts.min(axis=0)

        max_xyz = pts.max(axis=0)



        size = max_xyz - min_xyz





        print("\nComponent", i)



        print(" triangles:",

              cluster_n_triangles[i])



        print(" area:",

              cluster_area[i])



        print(" bbox size:",

              size)



        print(" center:",

              (min_xyz + max_xyz) / 2)





if __name__ == "__main__":



    if len(sys.argv) < 2:

        print(

            "Usage: python analyze_components.py xxx.ply"

        )

        exit()



    analyze_mesh(sys.argv[1])
