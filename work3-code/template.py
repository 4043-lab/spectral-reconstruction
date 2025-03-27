def set_template(args):
    # Set the templates here
    if args.template.find('my_model') >= 0:
        print("my_model")
        args.input_setting = 'H'
        args.input_mask = 'Phi_PhiPhiT'
        args.scheduler = 'CosineAnnealingLR'
        # args.milestones = [100, 150, 200]

    if args.template.find('dauhst') >= 0:
        print("dauhst")
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('specat') >= 0:
        print("specat")
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('bisrnet') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'