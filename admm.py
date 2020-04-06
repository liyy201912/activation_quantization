import numpy as np
import torch


def apply_quantization(args, model, name_list, device=None):
    for name, weight in model.named_parameters():
        if name in name_list:
            quantized_weight = weight.cpu().detach().numpy()
            if args.quant_type == "binary" or args.quant_type == "random_binary":
                quantized_weight[quantized_weight >= 0] = model.alpha[name]
                quantized_weight[quantized_weight < 0] = -model.alpha[name]
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "ternary":
                quantized_weight = np.where(quantized_weight > 0.5 * model.alpha[name], model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where(quantized_weight < -0.5 * model.alpha[name], -model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where(
                    (quantized_weight <= 0.5 * model.alpha[name]) & (quantized_weight >= -0.5 * model.alpha[name]), 0,
                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "fixed":
                Q = model.Q[name].cpu().detach()
                quantized_weight = model.alpha[name] * Q
                half_num_bits = args.num_bits - 1
                centroids = []
                for value in range(-2 ** half_num_bits - 1, 2 ** half_num_bits):
                    centroids.append(value)

                for i, value in enumerate(centroids):
                    if i == 0:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] < value + 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) - 1:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] >= value - 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    else:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= value - 0.5) & (
                                quantized_weight / model.alpha[name] < value + 0.5), value * model.alpha[name],
                                                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "one-side":
                quantized_weight = model.alpha[name] * Q
                half_num_bits = args.num_bits - 1
                centroids = []
                for value in range(-2 ** half_num_bits, 2 ** half_num_bits + 1):
                    if value == 0: continue
                    centroids.append(value)

                for i, value in enumerate(centroids):
                    if i == 0:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] < (value + 0.5),
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) - 1:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] >= (value - 0.5),
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) / 2 - 1:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= (value - 0.5)) & (
                                quantized_weight < 0), value * model.alpha[name],
                                                    quantized_weight)
                    elif i == len(centroids) / 2:
                        quantized_weight = np.where((quantized_weight >= 0) & (
                                quantized_weight / model.alpha[name] < (value + 0.5)), value * model.alpha[name],
                                                    quantized_weight)
                    else:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= (value - 0.5)) & (
                                quantized_weight / model.alpha[name] < (value + 0.5)), value * model.alpha[name],
                                                    quantized_weight)

                weight.data = torch.Tensor(quantized_weight).to(device)

            else:
                raise ValueError("Type '{}' is not supported quantized type!".format(args.quant_type))


def project_to_centroid(args, model, W, name, device, print):
    U = model.U[name].cpu().detach().numpy()
    Q = model.Q[name].cpu().detach().numpy()

    if args.quant_type == "binary":
        Q = np.where((W + U) >= 0, 1, -1)
        alpha = np.sum(np.multiply((W + U), Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ

    elif args.quant_type == "random_binary":
        p = np.where(1 < (W + 1) / 2, 1, (W + 1) / 2)
        p[p < 0] = 0
        bound = np.random.uniform(0, 1, p.shape)
        Q = np.where(bound < p, 1, -1)
        alpha = np.sum(np.multiply((W + U), Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ
    else:
        alpha = np.mean(np.abs(W[np.nonzero(W)]))  # initialize alpha
        # alpha = model.alpha[name]
        num_iter_quant = 6  # caffe code 20, in paper less than 5 iterations

    if args.quant_type == "ternary":
        for n in range(num_iter_quant):
            # pre_alpha = alpha
            Q = np.where((W + U) / alpha > 0.5, 1, Q)
            Q = np.where((W + U) / alpha < -0.5, -1, Q)
            Q = np.where(((W + U) / alpha >= -0.5) & ((W + U) / alpha <= 0.5), 0, Q)
            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.sum(Q ** 2)
            alpha /= QtQ
            # if args.verbose:
            #     print("at layer {}, alpha({})-alpha({}): {}".format(name, n, n - 1, alpha - pre_alpha))

    # 2bits [-1, 0, 1] 3bits [-3, -2, -1, 0, 1, 2, 3]
    elif args.quant_type == "fixed":
        half_num_bits = args.num_bits - 1
        centroids = []
        for value in range(-2 ** half_num_bits + 1, 2 ** half_num_bits):
            centroids.append(value)

        for n in range(num_iter_quant):
            Q = np.where(np.round((W + U) / alpha) <= centroids[0], centroids[0], Q)
            Q = np.where(np.round((W + U) / alpha) >= centroids[-1], centroids[-1], Q)
            Q = np.where((np.round((W + U) / alpha) < centroids[-1]) & (np.round((W + U) / alpha) > centroids[0]),
                         np.round((W + U) / alpha), Q)

            # for i, value in enumerate(centroids):
            #     if i == 0:
            #         Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
            #     elif i == len(centroids) - 1:
            #         Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
            #     else:
            #         Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value, Q)

            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.multiply(Q, Q)
            QtQ = np.sum(QtQ)
            alpha /= QtQ

    elif args.quant_type == "one-side":
        # half_num_bits = args.num_bits - 1
        # centroids = []
        # for value in range(-2 ** half_num_bits, 2 ** half_num_bits + 1):
        #     if value == 0: continue
        #     centroids.append(value)
        # for n in range(num_iter_quant):
        #     for i, value in enumerate(centroids):
        #         if i == 0:
        #             Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
        #         elif i == len(centroids) - 1:
        #             Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
        #         elif i == len(centroids) / 2 - 1:
        #             Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < 0), value, Q)
        #         elif i == len(centroids) / 2:
        #             Q = np.where((((W + U) / alpha) >= 0) & (((W + U) / alpha) < (value + 0.5)), value, Q)
        #         else:
        #             Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value, Q)
        # 
        #     alpha = np.sum(np.multiply((W + U), Q))
        #     QtQ = np.multiply(Q, Q)
        #     QtQ = np.sum(QtQ)
        #     alpha /= QtQ

        num_bits = args.num_bits  # not include zero, one side
        alpha = alpha / ((1 + 2 ** num_bits) * (2 ** (num_bits - 1)))
        centroids = []
        for value in range(-2 ** num_bits, 2 ** num_bits + 1):
            if value == 0: continue
            centroids.append(value)

        for n in range(num_iter_quant):
            for i, value in enumerate(centroids):
                if i == 0:
                    Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
                elif i == len(centroids) - 1:
                    Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
                elif i == len(centroids) / 2 - 1:
                    Q = np.where((((W + U) / alpha) >= (value - 0.5)) & ((W + U) < 0), value, Q)
                elif i == len(centroids) / 2:
                    Q = np.where(((W + U) >= 0) & ((W + U) / alpha < (value + 0.5)), value, Q)
                else:
                    Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value, Q)
            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.sum(Q ** 2)
            alpha /= QtQ

    model.U[name] = torch.Tensor(U).to(device)
    model.Q[name] = torch.Tensor(Q).to(device)
    model.Z[name] = alpha * model.Q[name]
    model.alpha[name] = alpha

    return model.Z[name], model.alpha[name], model.Q[name]


def admm_initialization(args, model, device, name_list, print):
    model.alpha = {}
    model.rhos = {}
    model.U = {}
    model.Q = {}  # alpha * Q = Z
    model.Z = {}

    for name, param in model.named_parameters():
        if name in name_list:
            print(name)
            model.rhos[name] = args.reg_lambda
            weight = param.cpu().detach().numpy()
            if args.quant_type == "binary" or args.quant_type == "ternary":
                model.alpha[name] = np.mean(np.abs(weight))  # initialize alpha
                # model.alpha[name] = np.mean(np.abs(weight[np.nonzero(weight)]))  # initialize alpha
            elif args.quant_type == "fixed":
                model.alpha[name] = np.mean(np.abs(weight[np.nonzero(weight)]))
                half_num_bits = args.num_bits - 1
                model.alpha[name] = model.alpha[name] / ((2 ** half_num_bits - 1) / 2)
                # model.alpha[name] = np.max(np.abs(weight))
                # half_num_bits = args.num_bits - 1
                # model.alpha[name] /= half_num_bits ** 2 - 1

            elif args.quant_type == "one-side":
                model.alpha[name] = 0  # initialize alpha
                # model.alpha[name] = np.mean(np.abs(weight[np.nonzero(weight)]))
                # half_num_bits = args.num_bits - 1
                # model.alpha[name] = model.alpha[name] / ((2 ** half_num_bits + 1) / 2)

            model.U[name] = torch.zeros(param.shape).to(device)
            model.Q[name] = torch.zeros(param.shape).to(device)
            updated_Z, updated_aplha, updated_Q = project_to_centroid(args, model, weight, name, device, print)
            model.Z[name] = updated_Z


def append_admm_loss(model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in model.rhos:
            continue
        admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z[name] + model.U[name], p=2) ** 2)

    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def z_u_update(args, model, device, name_list, print):
    print("Updating Z, U!!!!!!")
    for name, W in model.named_parameters():
        if name not in name_list:
            continue
        Z_prev = torch.clone(model.Z[name])
        weight = W.cpu().detach().numpy()
        updated_Z, updated_alpha, updated_Q = project_to_centroid(args, model, weight, name,
                                                                  device, print)  # equivalent to Euclidean Projection
        model.rhos[name] = model.rhos[name] * 1.5
        if model.rhos[name] >= 100:
            model.rhos[name] = 100
        # bound1 = torch.sqrt(torch.sum((W - model.Z[name]) ** 2)).item()
        # bound2 = torch.sqrt(torch.sum((model.Z[name] - Z_prev) ** 2)).item()
        # mu_value = 0.5
        # tau_value = 2  # 1+tau tau belongs (0,1)
        #
        # if mu_value * bound1 >= bound2:
        #     model.rhos[name] = model.rhos[name] * tau_value
        # elif bound1 < mu_value * bound2:
        #     model.rhos[name] = model.rhos[name] / tau_value
        # else:
        #     model.rhos[name] = model.rhos[name]

        if (args.verbose):
            print("at layer {}. W(k+1)-Z(k+1): {}".format(name, torch.sqrt(
                torch.sum((W - model.Z[name]) ** 2)).item()))
            print("at layer {}, Z(k+1)-Z(k): {}".format(name, torch.sqrt(
                torch.sum((model.Z[name] - Z_prev) ** 2)).item()))
            print("at layer {}, alpha(k+1):{}, rho(k+1):{}".format(name, model.alpha[name], model.rhos[name]))

        model.U[name] = W - model.Z[name] + model.U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def test_sparsity(model, print):
    """
         test sparsity for every involved layer and the overall compression rate
    """

    layer = 0
    tot_param = 0
    tot_zeros = 0
    for name, param in model.named_parameters():
        # if param.shape.__len__() is 4 and "shortcut" not in name:
        if param.shape.__len__() is 4:
            layer += 1
            num_tot = param.detach().cpu().numpy().size
            num_zeros = param.detach().cpu().eq(0).sum().item()
            sparsity = (num_zeros / num_tot) * 100
            density = 100 - sparsity
            tot_param += num_tot
            tot_zeros += num_zeros
            print("{}, {}, density: {:.2f}%, sparsity:{:.2f}%, total: {}, zeros: {}, non-zeros: {} ".format(layer, name,
                                                                                                            density,
                                                                                                            sparsity,
                                                                                                            num_tot,
                                                                                                            num_zeros,
                                                                                                            tot_param - tot_zeros))

    print(
        "Total parameters: {}, Total zero: {}, Total non-zeros: {}".format(tot_param, tot_zeros, tot_param - tot_zeros))
    print("Total sparsity: {:.4f}, compression rate: {:.4f}".format(tot_zeros / tot_param,
                                                                    tot_param / (tot_param - tot_zeros)))


def admm_adjust_learning_rate(optimizer, epoch, args):
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch

    """
    admm_epoch = args.admm_epochs
    init_lr = args.lr * (0.9 ** (epoch // 30))
    if epoch % admm_epoch == 0:
        lr = init_lr
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch // 3  # roughly every 1/3 admm_epoch.

        lr = init_lr * (0.5 ** (admm_epoch_offset // admm_step))

    print("learning rate =", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth=False):
    return lam * criterion(pred, y_a) + \
           (1 - lam) * criterion(pred, y_b)
