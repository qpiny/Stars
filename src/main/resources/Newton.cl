kernel void VectorAdd(
	global const float3* from_position,
	global const float3* from_speed,
	global const float* weight,
	global       float3* to_position,
	global       float3* to_speed,
	int numElements) {
		int from = get_global_id(0);
		to_speed[from] = from_speed[from];
		for (int to = 0; to < numElements; to++) {
			float r = distance(from_position[from], from_position[to]);
			float force = (weight[from] * weight[to]) / (r * r);
			float3 dir = normalize((float3)(to - from));
			to_speed[from] += dir * force;
		}
		to_position[from] = from_position[from] + to_speed[from];
}