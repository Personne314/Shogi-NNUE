
/**
 * @brief Perform one optimization step using Adam
 * @param weights Flat array of master weights to update (read/write).
 * @param gradients Flat array of accumulated gradients for the current batch (read/write — zeroed on exit).
 * @param m First moment (mean) buffer, one entry per parameter (read/write — persists across steps).
 * @param v Second moment (uncentered variance) buffer, one entry per parameter (read/write — persists across steps).
 * @param lr Learning rate.
 * @param beta1 Exponential decay rate for the first moment (typically 0.9).
 * @param beta2 Exponential decay rate for the second moment (typically 0.999).
 * @param eps Numerical stability term added to the denominator (typically 1e-8).
 * @param timestep Current optimization step t, used for bias correction (≥ 1).
 * @param total_params Total number of scalar parameters across all arrays.
 */
__kernel void adam_update(
	__global float *weights,
	__global float *gradients,
	__global float *m,
	__global float *v,
	const float lr,
	const float beta1,
	const float beta2,
	const float eps,
	const int timestep,
	const int total_params)
{
	int gid = get_global_id(0);
	if (gid >= total_params) return;

	float g = gradients[gid];
	float m_prev = m[gid];
	float v_prev = v[gid];

	// Compute the averages.
	float m_t = beta1 * m_prev + (1.0f - beta1) * g;
	float v_t = beta2 * v_prev + (1.0f - beta2) * g * g;
	m[gid] = m_t;
	v[gid] = v_t;

	// Adam bias correction.
	float beta1_t = 1.0f - pow(beta1, (float)timestep);
	float beta2_t = 1.0f - pow(beta2, (float)timestep);
	float m_hat = m_t / beta1_t;
	float v_hat = v_t / beta2_t;

	// Weight update.
	weights[gid] -= lr * m_hat / (sqrt(v_hat) + eps);

	// Gradient reset.
	gradients[gid] = 0.0f;

}
