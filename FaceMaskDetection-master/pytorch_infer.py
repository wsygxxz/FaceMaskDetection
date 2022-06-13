# -*- coding:utf-8 -*-
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

# model = load_pytorch_model('models/face_mask_detection.pth');
model = load_pytorch_model('models/model360.pth')
# anchor configuration
# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
# 横着增加一维
id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):

    # conf_thresh  ，置信度阈值，置信度低于该阈值的预测框会被抛弃。就是检测物体框的置信度，它包含物体的概率或分数
    # Nms_thresh, 就是极大值抑制里面的一个参数，主要用于减少重复预测框的数量
    #
    # iou_thresh,  就是预测和GT两个框的交并比参数，高于就代表有正，低于代表负


    '''
    Main function of detection inference
    :param image: 3D numpy array of image，3D numpy图片数组
    :param conf_thresh: the min threshold of classification probabity.：分类概率的最小阈值。
    :param iou_thresh: the IOU threshold of NMS，网管的IOU门限
    :param target_shape: the model input size.模型输入大小
    :param draw_result: whether to daw bounding box to the image.是否将边框拖入图像
    :param show_result: whether to display the image.是否显示图像
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    # 输入图片
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    # np.expand_dims:用于扩展数组的形状
    # np.expand_dims(a, axis=0) 表示在0位置添加数据, 转换结果如下：
    # In [13]:
    # b = np.expand_dims(a, axis=0)
    # b
    # Out[13]:
    # array([[[[1, 2, 3],
    #          [4, 5, 6]]]])
    # In [14]:
    # b.shape
    # Out[14]:
    # (1, 1, 2, 3)
    #
    # np.expand_dims(a, axis=1)表示在1位置添加数据,转换结果如下：
    # In [15]:
    # c = np.expand_dims(a, axis=1)
    # c
    # Out[15]:
    # array([[[[1, 2, 3],
    #          [4, 5, 6]]]])
    #
    # In [16]:
    # c.shape
    # Out[16]:
    # (1, 1, 2, 3)
    # np.expand_dims(a, axis=3)表示在3位置添加数据,转换结果如下
    # In [19]:
    # e = np.expand_dims(a, axis=3)
    # e
    #
    # In [20]:
    # e.shape
    # Out[20]:
    # (1, 2, 3, 1)

    image_transposed = image_exp.transpose((0, 3, 1, 2))
    # images = raw_float.reshape([-1, num_channels, img_size, img_size])
    # images = images.transpose([0, 2, 3, 1])
    # 这里images.transpose([0, 2, 3, 1])是将images的shape变成[num,img_size, img_size,num_channels ]


    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    # 按照行索引最大值
    #
    # 1.对一个一维向量
    #
    # import numpy as np
    # a = np.array([3, 1, 2, 4, 6, 1])
    # b=np.argmax(a)#取出a中元素最大值所对应的索引，此时最大值位6，其对应的位置索引值为4，（索引值默认从0开始）
    # print(b)#4
    # keep_idx is the alive bounding box after nms.


    # 2.对2维向量（通常意义下的矩阵）a[][]
    # import numpy as np
    # a = np.array([[1, 5, 5, 2],
    #               [9, 6, 2, 8],
    #               [3, 7, 9, 1]])
    # b=np.argmax(a, axis=0)#对二维矩阵来讲a[0][1]会有两个索引方向，第一个方向为a[0]，默认按列方向搜索最大值
    # #a的第一列为1，9，3,最大值为9，所在位置为1，
    # #a的第一列为5，6，7,最大值为7，所在位置为2，
    # #此此类推，因为a有4列，所以得到的b为1行4列，
    # print(b)#[1 2 2 1]
    #
    # c=np.argmax(a, axis=1)#现在按照a[0][1]中的a[1]方向，即行方向搜索最大值，
    # #a的第一行为1，5，5，2,最大值为5（虽然有2个5，但取第一个5所在的位置），索引值为1，
    # #a的第2行为9，6，2，8,最大值为9，索引值为0，
    # #因为a有3行，所以得到的c有3个值，即为1行3列
    # print(c)#[1 0 2]

    # keep_idx是nms之后的活动边界框。
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        #  裁剪坐标，避免该值超出图像边界
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            #     0 戴口罩 绿色
            else:
                color = (255, 0, 0)
            #     1 不带  红色
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
        # 改为以下
        # image6 = Image.fromarray(image).show()
        # cv2.imwrite('1.png', image6)

        # cv2.imshow('image', image6)
        # image.show()
    return output_info
# Image.fromarray的作用：
# 简而言之，就是实现array到image的转换

def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # MPEG-4编码 .mp4 可指定结果视频的大小
    # cv2.VideoWriter_fourcc('X','2','6','4')
    # MPEG-4编码 .mp4 可指定结果视频的大小
    # cv2.VideoWriter_fourcc('I', '4', '2', '0')
    # 该参数是YUV编码类型，文件名后缀为.avi 广泛兼容，但会产生大文件
    # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I')
    # 该参数是MPEG-1编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 该参数是MPEG-4编码类型，文件名后缀为.avi，可指定结果视频的大小
    # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O')
    # 该参数是Ogg Vorbis,文件名后缀为.ogv
    # cv2.VideoWriter_fourcc('F', 'L', 'V', '1')
    # 该参数是Flash视频，文件名后缀为.flv

    writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            # cv2.imwrite('1.png', img)

            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    # add_argument()是用来指定程序需要接受的命令参数，其中input_dir这一类直接用单引号括起来的表示定位参数而margin这一类前加有–为可选参数，并规定定位参数必选，可选参数可选。
    parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    # parser.add_argument('--img-path', type=str, default='img/3.png', help='path to your image.')
    parser.add_argument('--img-path', type=str, default='img/5.jpeg', help='path to your image.')
    # parser.add_argument('--img-path', type=str, default='img/6.jpeg', help='path to your image.')
    # parser.add_argument('--img-path', type=str, default='img/9.jpeg', help='path to your image.')
    # parser.add_argument('--img-path', type=str, default='img/10.jpeg', help='path to your image.')
    # parser.add_argument('--img-path', type=str, default='img/11.jpeg', help='path to your image.')
    # parser.add_argument('--img-path', type=str, default='img/3.jpeg', help='path to your image.')


    # parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--video-path', type=str, default='D:\py\FaceMaskDetection-master\img/5.mp4', help='path to your video, `0` means to use camera.')
    parser.add_argument('--video-path', type=str, default='D:\py\FaceMaskDetection-master\img/6.mp4', help='path to your video, `0` means to use camera.')

    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, target_shape=(360, 360))

    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)
