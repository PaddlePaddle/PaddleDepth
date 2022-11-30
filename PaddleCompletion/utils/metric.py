import paddle
import math
import numpy as np

lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return paddle.log(x) / lg_e_10


class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
        self.photometric = 0

        self.irmse1 = 0
        self.mse1 = 0
        self.imae1 = 0
        self.rmse1 = 0
        self.mae1 = 0

        self.irmse2 = 0
        self.mse2 = 0
        self.imae2 = 0
        self.rmse2 = 0
        self.mae2 = 0


    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

        self.irmse1 = np.inf
        self.mse1 = np.inf
        self.imae1 = np.inf
        self.rmse1 = np.inf
        self.mae1 = np.inf

        self.irmse2 = np.inf
        self.mse2 = np.inf
        self.imae2 = np.inf
        self.rmse2 = np.inf
        self.mae2 = np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, \
            delta1, delta2, delta3, gpu_time, data_time, irmse1, imae1, mse1, rmse1, mae1, \
               irmse2, imae2, mse2, rmse2, mae2, silog, photometric=0):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.photometric = photometric

        self.irmse1 = irmse1
        self.imae1 = imae1
        self.mse1 = mse1
        self.rmse1 = rmse1
        self.mae1 = mae1

        self.irmse2 = irmse2
        self.imae2 = imae2
        self.mse2 = mse2
        self.rmse2 = rmse2
        self.mae2 = mae2

    def evaluate(self, output, target, depth_output, lidar_output, photometric=0):
        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((paddle.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm)**2).mean())

        maxRatio = paddle.max((output_mm / target_mm, target_mm / output_mm))
        self.delta1 = float(paddle.to_tensor((maxRatio < 1.25),dtype="float32").mean())
        self.delta2 = float(paddle.to_tensor((maxRatio < 1.25**2),dtype="float32").mean())
        self.delta3 = float(paddle.to_tensor((maxRatio < 1.25**3),dtype="float32").mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = paddle.log(target[valid_mask]) - paddle.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log -
                               log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask])**(-1)
        inv_target_km = (1e-3 * target[valid_mask])**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((paddle.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.photometric = float(photometric)


        output_mm = 1e3 * depth_output[valid_mask]
        abs_diff = (output_mm - target_mm).abs()
        self.mse1 = float((paddle.pow(abs_diff, 2)).mean())
        self.rmse1 = math.sqrt(self.mse1)
        self.mae1 = float(abs_diff.mean())
        # convert from meters to km
        inv_output_km = (1e-3 * depth_output[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse1 = math.sqrt((paddle.pow(abs_inv_diff, 2)).mean())
        self.imae1 = float(abs_inv_diff.mean())


        output_mm = 1e3 * lidar_output[valid_mask]
        abs_diff = (output_mm - target_mm).abs()
        self.mse2 = float((paddle.pow(abs_diff, 2)).mean())
        self.rmse2 = math.sqrt(self.mse2)
        self.mae2 = float(abs_diff.mean())
        # convert from meters to km
        inv_output_km = (1e-3 * lidar_output[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse2 = math.sqrt((paddle.pow(abs_inv_diff, 2)).mean())
        self.imae2 = float(abs_inv_diff.mean())

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count_ = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0

        self.sum_irmse1 = 0
        self.sum_imae1 = 0
        self.sum_mse1 = 0
        self.sum_rmse1 = 0
        self.sum_mae1 = 0

        self.sum_irmse2 = 0
        self.sum_imae2 = 0
        self.sum_mse2 = 0
        self.sum_rmse2 = 0
        self.sum_mae2 = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count_ += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

        self.sum_irmse1 += n * result.irmse1
        self.sum_imae1 += n * result.imae1
        self.sum_mse1 += n * result.mse1
        self.sum_rmse1 += n * result.rmse1
        self.sum_mae1 += n * result.mae1

        self.sum_irmse2 += n * result.irmse2
        self.sum_imae2 += n * result.imae2
        self.sum_mse2 += n * result.mse2
        self.sum_rmse2 += n * result.rmse2
        self.sum_mae2 += n * result.mae2

    def average(self):
        avg = Result()
       
        if self.count_ > 0:
            avg.update(
                self.sum_irmse / self.count_, self.sum_imae / self.count_,
                self.sum_mse / self.count_, self.sum_rmse / self.count_,
                self.sum_mae / self.count_, self.sum_absrel / self.count_,
                self.sum_squared_rel / self.count_, self.sum_lg10 / self.count_,
                self.sum_delta1 / self.count_, self.sum_delta2 / self.count_,
                self.sum_delta3 / self.count_, self.sum_gpu_time / self.count_,
                self.sum_data_time / self.count_,
                self.sum_irmse1 / self.count_, self.sum_imae1 / self.count_,
                self.sum_mse1 / self.count_, self.sum_rmse1 / self.count_,
                self.sum_mae1 / self.count_,
                self.sum_irmse2 / self.count_, self.sum_imae2 / self.count_,
                self.sum_mse2 / self.count_, self.sum_rmse2 / self.count_,
                self.sum_mae2 / self.count_,
                self.sum_silog / self.count_,
                self.sum_photometric / self.count_)
        return avg
