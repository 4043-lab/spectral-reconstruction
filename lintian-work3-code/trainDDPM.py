import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import wandb
import utils
import random
from model.sr3_modules import transformer
from cassi_utils import *
from architecture import *
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/cassi.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_id', type=str, default='6')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-cassi_model', '--cassi_model', default='dauhst')
    parser.add_argument('-pretrained_model_path', '--pretrained_model_path', default='experiments/dauhst_3stg_icvl/model_epoch_230.pth')



    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # dataset，当args.phase=train时，for循环同时加载trainSet和valSet，当args.phase=val时，只加载valSet
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            print(val_set)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    cassi_model = model_generator(None, args.cassi_model, args.pretrained_model_path).cuda()
    # cassi_model = dataparallel(cassi_model, args)
    # for k, v in cassi_model.named_parameters():
    #     v.requires_grad = False
    cassi_model.eval()

    # degradation predict model
    # model_restoration = transformer.Uformer()
    # model_restoration.cuda()
    # path_chk_rest_student = '../models/model_epoch_450.pth'
    # utils.load_checkpoint(model_restoration, path_chk_rest_student)
    # model_restoration.eval()

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                meas, Phi, Phi_s = Variable(
                    train_data['meas']).cuda(), Variable(train_data['Phi']).cuda(), Variable(train_data['Phi_s']).cuda()

                with torch.no_grad():
                    train_data['SR'] = cassi_model(meas, Phi, Phi_s).detach().cpu()

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        # print(val_data['HR'].shape)
                        meas, Phi, Phi_s = Variable(
                            val_data['meas']).cuda(), Variable(val_data['Phi']).cuda(), Variable(
                            val_data['Phi_s']).cuda()

                        with torch.no_grad():
                            val_data['SR'] = cassi_model(meas, Phi, Phi_s).detach().cpu()
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        update_mask = Metrics.tensor2img(visuals['UM'])
                        # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        # fake_img = Metrics.tensor2img(visuals['INF'], min_max=(0, 1))  # uint8

                        # generation
                        # Metrics.save_mat(
                        #     hr_img, '{}/{}_{}_hr.mat'.format(result_path, current_step, idx))
                        Metrics.save_mat(
                            sr_img, '{}/{}_{}_sr.mat'.format(result_path, current_step, idx))
                        Metrics.save_mat(
                            update_mask, '{}/{}_{}_um.mat'.format(result_path, current_step, idx))
                        # Metrics.save_mat(
                        #     lr_img, '{}/{}_{}_lr.mat'.format(result_path, current_step, idx))
                        # Metrics.save_mat(
                        #     fake_img, '{}/{}_{}_inf.mat'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        # print("hr_img.shape：{}".format(hr_img.shape))
                        # print("sr_img.shape：{}".format(sr_img.shape))
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)

                        # if wandb_logger:
                        #     wandb_logger.log_image(
                        #         f'validation_{idx}',
                        #         np.concatenate((sr_img, hr_img), axis=1)
                        #     )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()
            # x_hat = model_restoration((val_data['SR'] + 1) / 2, val_data['mask'])
            # x_hat = torch.clamp(x_hat, 0, 1)
            # # x_hat = x_hat * 2 - 1
            # h_hat = (val_data['SR']+1) / (2 *(x_hat) + 1e-4)
            # h_hat = torch.clamp(h_hat, 0, 1)
            # h_hat = torch.where(h_hat == 0, h_hat + 1e-4, h_hat)
            # diffusion.test(h_hat, continous=True)
            # visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            res = Metrics.tensor2img(visuals['SR'][-1])
            avg_channel = np.mean(res, axis=(0, 1))
            avg_channel_gt = np.mean(hr_img, axis=(0, 1))
            # res = res * avg_channel_gt / avg_channel
            eval_psnr = Metrics.calculate_psnr(res, hr_img)
            eval_ssim = Metrics.calculate_ssim(res, hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            print(f"ID: {idx}; PSNR: {eval_psnr}; SSIM: {eval_ssim}")

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
