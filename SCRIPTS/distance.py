import geopy.distance

k_coord = (40.741944, -74.179167)
m_coord = (41.751416, -89.616813)
w_coord = (40.675556, -105.039167)

print(f"WWV to K2MFF {geopy.distance.geodesic(w_coord, k_coord).km}")
print(f"WWV to Midpoint {geopy.distance.geodesic(w_coord, m_coord).km}")
