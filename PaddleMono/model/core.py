from model.mldanet.layers import *
from model import monodepthv2, mldanet, bts

def build_model(opt):
    models = {}
    parameters_to_train = []
    num_input_frames = len(opt.frame_ids)
    num_pose_frames = 2 if opt.pose_model_input == "pairs" else num_input_frames

    if opt.type == "MLDANet":
        models["encoder"] = mldanet.ResnetEncoder_multi_sa_add_reduce_640(
            opt.num_layers, opt.weights_init)
        parameters_to_train += list(models["encoder"].parameters())

        models["depth"] = mldanet.DepthDecoderAttention_edge(
            models["encoder"].num_ch_enc, opt.scales)
        parameters_to_train += list(models["depth"].parameters())

        models["pose_encoder"] = mldanet.ResnetEncoder_multi_sa_add_reduce_640(
            opt.num_layers,
            opt.weights_init,
            num_input_images=num_pose_frames)
        parameters_to_train += list(models["pose_encoder"].parameters())

        models["pose"] = mldanet.PoseDecoder(
            models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        parameters_to_train += list(models["pose"].parameters())

    elif opt.type == "MonoDepthv2":
        if opt.use_stereo:
            opt.frame_ids.append("s")

        models["encoder"] = monodepthv2.ResnetEncoder(
            opt.num_layers, opt.weights_init)
        parameters_to_train += list(models["encoder"].parameters())

        models["depth"] = monodepthv2.DepthDecoder(
            models["encoder"].num_ch_enc, opt.scales)
        parameters_to_train += list(models["depth"].parameters())

        if not (opt.use_stereo and opt.frame_ids == [0]):
            if opt.pose_model_type == "separate_resnet":
                models["pose_encoder"] = monodepthv2.ResnetEncoder(
                    opt.num_layers,
                    opt.weights_init,
                    num_input_images=num_pose_frames)

                parameters_to_train += list(models["pose_encoder"].parameters())

                models["pose"] = monodepthv2.PoseDecoder(
                    models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif opt.pose_model_type == "shared":
                models["pose"] = monodepthv2.PoseDecoder(
                    models["encoder"].num_ch_enc, num_pose_frames)

            elif opt.pose_model_type == "posecnn":
                models["pose"] = monodepthv2.PoseCNN(
                    num_input_frames if opt.pose_model_input == "all" else 2)

            parameters_to_train += list(models["pose"].parameters())

        if opt.predictive_mask:
            assert opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            models["predictive_mask"] = monodepthv2.DepthDecoder(
                models["encoder"].num_ch_enc, opt.scales,
                num_output_channels=(len(opt.frame_ids) - 1))
            parameters_to_train += list(models["predictive_mask"].parameters())

    elif opt.type == "BTS":
        models["encoder"] = bts.encoder(opt)
        parameters_to_train += list(models["encoder"].parameters())
        models["depth"] = bts.bts(opt, models["encoder"].feat_out_channels)
        parameters_to_train += list(models["depth"].parameters())

    return models, parameters_to_train
