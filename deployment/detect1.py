import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str, default='cfg/yolo-fastest.cfg', help='*.cfg path')
# parser.add_argument('--names', type=str, default='data/face_mask.names', help='*.names path')
# parser.add_argument('--weights', type=str, default='weights/best.weights', help='weights path')
# parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
# parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
# parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
# parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
# parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
# parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
# parser.add_argument('--view-img', action='store_true', help='display results')
# parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# opt = parser.parse_args()

def make_outfolder(path):
    if (pth:=os.path.dirname(path)) != "":
        os.makedirs(pth, exist_ok=True)

def detect(save_img=True, out="static/images/test.jpg", source='data/samples/good_test.jpg', no_save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half, view_img, save_txt = opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     # print(f"{out}")
    #     shutil.rmtree(os.path.dirname(out))  # delete output folder
    make_outfolder(out)
    # make_outfolder(out_csv)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = not no_save_img
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0, 69, 255], [170, 178, 32]] # Color for binary classification in B,G,R

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        imgname = path.split('/')[-1]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #save_path = str(Path(out) / Path(p).name)
            save_path = str(Path(out) / 'test.jpg')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            label = None
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
#                for c in det[:, -1].unique():
#                    n = (det[:, -1] == c).sum()  # detections per class
#                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(f"{imgname},{cls},{xywh[0]},{xywh[1]},{xywh[2]},{xywh[3]}\n")
                            # file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            print(label)
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolo-fastest.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/face_mask.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best.weights', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument("--no_save_img", action="store_true", default=False, help="do not save images")
    parser.add_argument('--output', type=str, default='output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect(out=opt.output, source=opt.source, no_save_img=opt.no_save_img)
