#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

/**
 * @struct TrainingData
 * @brief Contains the description on one board state for training.
 */
typedef struct
{
	int features_current[80];
	int features_other[80];
	int num_current;
	int num_other;
	int score;
	float game_result;
} TrainingData;

/**
 * @brief Thread-safe float += operator.
 */
inline void atomicAddFloat(volatile __global float *addr, float val)
{
	union { unsigned int u32; float f32; } next, expected, current;
	current.f32 = *addr;
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

/**
 * @brief Sigmoid function.
 */
inline float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

/**
 * @brief Clipped ReLU for int values.
 */
inline float d_clipped_relu_int(int raw_sum, int shift)
{
	int val = raw_sum >> shift;
	return (val > 0 && val < 127) ? 1.0f : 0.0f;
}

/**
 * @brief Pre-calculates the quantized weights for the entire network once per batch.
 */
__kernel void quantize_weights(
	__global const float* weights,
	__global int* q_weights,
	const int off_in_w,  const int off_in_b,
	const int off_fc1_w, const int off_fc1_b,
	const int off_fc2_w, const int off_fc2_b,
	const int off_out_w, const int off_out_b,
	const int total_params
) {
	int gid = get_global_id(0);
	if (gid >= total_params) return;

	float w = weights[gid];

	// Input Layer: scale 127
	if (gid >= off_in_w && gid < off_fc1_w) {
		q_weights[gid] = (int)round(w * 127.0f);
	}
	// FC1 & FC2 Weights & Output Weights: scale 64
	else if ((gid >= off_fc1_w && gid < off_fc1_b) || 
			 (gid >= off_fc2_w && gid < off_fc2_b) ||
			 (gid >= off_out_w && gid < off_out_b)) {
		q_weights[gid] = (int)round(w * 64.0f);
	}
	// Biases for Hidden and Output Layers: scale 64 * 127
	else if ((gid >= off_fc1_b && gid < off_fc2_w) || 
			 (gid >= off_fc2_b && gid < off_out_w) || 
			 gid == off_out_b) {
		q_weights[gid] = (int)round(w * (64.0f * 127.0f));
	}
}

/**
 * @brief Perform a combined forward and backward pass over one batch of training samples.
 */
__kernel void forward_backward(
	__global const TrainingData* batch,
	__global const float* weights,
	__global const int* q_weights,
	__global float* gradients,
	const int off_in_w,  const int off_in_b,
	const int off_fc1_w, const int off_fc1_b,
	const int off_fc2_w, const int off_fc2_b,
	const int off_out_w, const int off_out_b,
	const int batch_size,
	const float lambda,
	const float scale,
	__global float* batch_loss
) {
	int gid = get_global_id(0);
	if (gid >= batch_size) return;
	__global const TrainingData* data = &batch[gid];
	
	__global const float* out_w = weights + off_out_w;
	__global const float* fc2_w = weights + off_fc2_w;
	__global const float* fc1_w = weights + off_fc1_w;

	__global const int* q_in_w  = q_weights + off_in_w;
	__global const int* q_in_b  = q_weights + off_in_b;
	__global const int* q_fc1_w = q_weights + off_fc1_w;
	__global const int* q_fc1_b = q_weights + off_fc1_b;
	__global const int* q_fc2_w = q_weights + off_fc2_w;
	__global const int* q_fc2_b = q_weights + off_fc2_b;
	__global const int* q_out_w = q_weights + off_out_w;
	__global const int* q_out_b = q_weights + off_out_b;

	__global float* grad_in_w  = gradients + off_in_w;
	__global float* grad_in_b  = gradients + off_in_b;
	__global float* grad_fc1_w = gradients + off_fc1_w;
	__global float* grad_fc1_b = gradients + off_fc1_b;
	__global float* grad_fc2_w = gradients + off_fc2_w;
	__global float* grad_fc2_b = gradients + off_fc2_b;
	__global float* grad_out_w = gradients + off_out_w;
	__global float* grad_out_b = gradients + off_out_b;

	int acc_current[256];
	int acc_other[256];
	for (int i = 0; i < 256; ++i) {
		acc_current[i] = q_in_b[i];
		acc_other[i]   = q_in_b[i];
	}

	for (int f = 0; f < data->num_current; ++f) {
		int idx = data->features_current[f] * 256;
		for (int i = 0; i < 256; ++i) acc_current[i] += q_in_w[idx + i];
	}
	for (int f = 0; f < data->num_other; ++f) {
		int idx = data->features_other[f] * 256;
		for (int i = 0; i < 256; ++i) acc_other[i] += q_in_w[idx + i];
	}

	int clip_in[512];
	for (int i = 0; i < 256; ++i) {
		clip_in[i]       = clamp(acc_current[i], 0, 127);
		clip_in[i + 256] = clamp(acc_other[i], 0, 127);
	}

	int fc1_raw[32];
	int fc1_out[32];
	for (int i = 0; i < 32; ++i) {
		int sum = q_fc1_b[i];
		int w_offset = i * 512;
		for (int j = 0; j < 512; ++j) sum += q_fc1_w[w_offset + j] * clip_in[j];
		fc1_raw[i] = sum;
		fc1_out[i] = clamp(sum >> 6, 0, 127);
	}

	int fc2_raw[32];
	int fc2_out[32];
	for (int i = 0; i < 32; ++i) {
		int sum = q_fc2_b[i];
		int w_offset = i * 32;
		for (int j = 0; j < 32; ++j) sum += q_fc2_w[w_offset + j] * fc1_out[j];
		fc2_raw[i] = sum;
		fc2_out[i] = clamp(sum >> 6, 0, 127);
	}

	int score_int = q_out_b[0];
	for (int j = 0; j < 32; ++j) score_int += q_out_w[j] * fc2_out[j];

	float score_pred = (float)score_int / 16.0f;
	float p_net = sigmoid(score_pred / scale);
	float p_tree = sigmoid((float)data->score / scale);
	float target = lambda * data->game_result + (1.0f - lambda) * p_tree;
	
	float d_out = 2.0f * (p_net - target) * p_net * (1.0f - p_net) / scale;
	d_out /= (float)batch_size;
	
	float error = p_net - target;
	float mse = error * error;
	atomicAddFloat(batch_loss, mse);

	atomicAddFloat(&grad_out_b[0], d_out);
	for (int j = 0; j < 32; ++j) {
		atomicAddFloat(&grad_out_w[j], d_out * (float)fc2_out[j]);
	}

	float d_fc2[32];
	for (int i = 0; i < 32; ++i) {
		d_fc2[i] = d_out * out_w[i] * d_clipped_relu_int(fc2_raw[i], 6);
		atomicAddFloat(&grad_fc2_b[i], d_fc2[i]);
		int w_offset = i * 32;
		for (int j = 0; j < 32; ++j) {
			atomicAddFloat(&grad_fc2_w[w_offset + j], d_fc2[i] * (float)fc1_out[j]);
		}
	}

	float d_fc1[32];
	for (int i = 0; i < 32; ++i) {
		float err = 0.0f;
		for (int k = 0; k < 32; ++k) err += d_fc2[k] * fc2_w[k * 32 + i];
		d_fc1[i] = err * d_clipped_relu_int(fc1_raw[i], 6);
		atomicAddFloat(&grad_fc1_b[i], d_fc1[i]);
		int w_offset = i * 512;
		for (int j = 0; j < 512; ++j) {
			atomicAddFloat(&grad_fc1_w[w_offset + j], d_fc1[i] * (float)clip_in[j]);
		}
	}

	float d_in[512];
	for (int j = 0; j < 512; ++j) {
		float err = 0.0f;
		for (int k = 0; k < 32; ++k) err += d_fc1[k] * fc1_w[k * 512 + j];
		int acc_val = (j < 256) ? acc_current[j] : acc_other[j - 256];
		d_in[j] = err * d_clipped_relu_int(acc_val, 0); 
	}
	
	for (int i = 0; i < 256; ++i) {
		atomicAddFloat(&grad_in_b[i], d_in[i] + d_in[i + 256]);
	}

	for (int f = 0; f < data->num_current; ++f) {
		int idx = data->features_current[f] * 256;
		for (int i = 0; i < 256; ++i) atomicAddFloat(&grad_in_w[idx + i], d_in[i]);
	}
	for (int f = 0; f < data->num_other; ++f) {
		int idx = data->features_other[f] * 256;
		for (int i = 0; i < 256; ++i) atomicAddFloat(&grad_in_w[idx + i], d_in[i + 256]);
	}
}
