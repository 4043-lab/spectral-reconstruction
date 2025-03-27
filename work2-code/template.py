def set_template(args):
    # Set the templates here
    if args.template.find('my_model') >= 0:
        print("my_model")
        args.input_setting = 'H'
        args.input_mask = 'Mask'
