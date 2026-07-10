"""
将预测结果转换为Waymo Open Dataset E2E Driving挑战提交格式
"""
import json
import re
import numpy as np
import math
import os
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2


def interpolate_trajectory(traj_input, input_hz=2, output_hz=4, duration=5.0):
    """
    将低频率轨迹插值成4Hz的轨迹
    在开头添加(0,0)作为0s时刻的起点，以便正确插值

    Args:
        traj_input: 输入轨迹点，shape [N, 2]
        input_hz: 输入轨迹的频率 (1, 2)
        output_hz: 输出轨迹的频率 (固定为4)
        duration: 轨迹总时长（秒），默认5秒

    Returns:
        traj_4hz: shape [20, 2], 4Hz频率的轨迹点 (对应0.25s, 0.5s, ..., 5.0s)
    """
    traj_input = np.array(traj_input)
    dt_input = 1.0 / input_hz  # 输入时间间隔
    dt_output = 1.0 / output_hz  # 输出时间间隔（0.25s）

    # 输入点数
    num_input_points = traj_input.shape[0]
    # 输出点数 (5秒 * 4Hz = 20个点，对应0.25s, 0.5s, ..., 5.0s)
    num_output_points = int(duration * output_hz)

    # 在开头添加(0, 0)作为0s时刻的起点
    traj_with_origin = np.vstack([[0.0, 0.0], traj_input])

    # 输入时间点：0.0, dt_input, 2*dt_input, ..., num_input_points*dt_input
    time_input = np.arange(num_input_points + 1) * dt_input

    # 目标时间点：0.25, 0.5, 0.75, ..., 5.0 (20个点)
    time_output = np.arange(1, num_output_points + 1) * dt_output

    # 对x和y分别进行线性插值
    traj_output = np.zeros((num_output_points, 2))
    traj_output[:, 0] = np.interp(time_output, time_input, traj_with_origin[:, 0])
    traj_output[:, 1] = np.interp(time_output, time_input, traj_with_origin[:, 1])

    return traj_output


def convert_predictions_to_submission(
    input_json_path,
    output_dir,
    num_shards=10,
    authors=['Author1', 'Author2'],
    affiliation='Your Affiliation',
    account_name='your_email@domain.com',
    method_name='YourMethodName',
    method_link='https://your_method_link',
    description='Method description',
    uses_public_model_pretraining=True,
    public_model_names=['Model1'],
    num_model_parameters='200M'
):
    """
    将预测结果JSON文件转换为Waymo提交格式

    Args:
        input_json_path: 输入的JSON文件路径
        output_dir: 输出目录
        num_shards: 分片数量
        其他参数: 提交信息
    """
    print(f"读取预测结果: {input_json_path}")

    # 根据文件后缀决定读取方式
    if input_json_path.endswith('.jsonl'):
        # JSONL格式：每行含 sample_id；轨迹优先从 generation_turn2 解析，否则从 generation
        pattern = r'\[([\d.-]+),\s*([\d.-]+)\]'
        predictions_data = []
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample_id = data.get('sample_id', '')
                    turn2 = (data.get('generation_turn2') or '').strip()
                    generation_text = (data.get('generation') or '').strip()
                    # 多轮输出：仅用第二段轨迹；无 generation_turn2 时沿用整段 generation（兼容旧格式）
                    traj_source = turn2 if turn2 else generation_text
                    matches = re.findall(pattern, traj_source)
                    pred_future = [[float(x), float(y)] for x, y in matches]
                    predictions_data.append({'sample_id': sample_id, 'pred_future': pred_future})
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效JSON行: {line[:50]}... 错误: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 处理行时出错: {e}")
                    continue
    else:
        # 普通JSON格式：直接读取
        with open(input_json_path, 'r') as f:
            predictions_data = json.load(f)

    print(f"总共有 {len(predictions_data)} 个预测样本")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 转换每个预测
    frame_predictions = []

    for i, pred_data in enumerate(predictions_data):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(predictions_data)}")

        # 提取sample_id作为frame_name
        frame_name = pred_data['sample_id']

        # 提取预测轨迹
        pred_future = pred_data['pred_future']
        pred_traj = np.asarray(pred_future, dtype=np.float32)
        if pred_traj.ndim != 2 or pred_traj.shape[1] != 2:
            raise ValueError(
                f"pred_future shape不符合预期: sample_id={frame_name}, "
                f"got shape={getattr(pred_traj, 'shape', None)}"
            )

        # 根据点数决定是否需要插值
        # print(pred_traj)
        pred_traj = pred_traj[:5, :]
        num_points = pred_traj.shape[0]

  
        # num_points = 5
        if num_points < 5:
            print("ori pred_traj", pred_traj)
            if num_points == 0:
                raise ValueError(f"pred_future为空，无法外推: sample_id={frame_name}")
            # 1Hz 五个点对应 1s~5s；外推时在逻辑上前面补 (0,0) 作为 0s，仅用相邻差分算步长，不把 (0,0) 写入输出
            origin = np.zeros((1, 2), dtype=pred_traj.dtype)
            work_traj = np.vstack([origin, pred_traj])
            step = work_traj[-1] - work_traj[-2]
            n_extra = 5 - num_points
            extra_idx = np.arange(1, n_extra + 1, dtype=pred_traj.dtype)[:, None]
            extrapolated_traj = work_traj[-1] + extra_idx * step
            pred_traj = np.vstack([pred_traj, extrapolated_traj])
            num_points = pred_traj.shape[0]
            if i < 3:
                print(
                    f"[debug extrap] sample_id={frame_name}, 少于5点，用(0,0)链上步长外推, "
                    f"extrap shape={extrapolated_traj.shape}"
                )
            print("after extrapolated pred_traj", pred_traj)
        if num_points == 5:
            # 1Hz -> 4Hz (5个点对应1s, 2s, 3s, 4s, 5s)
            pred_traj_4hz = interpolate_trajectory(pred_traj, input_hz=1, output_hz=4)
        elif num_points == 10:
            # 2Hz -> 4Hz (10个点对应0.5s, 1.0s, ..., 5.0s)
            pred_traj_4hz = interpolate_trajectory(pred_traj, input_hz=2, output_hz=4)
        elif num_points == 20:
            # 已经是4Hz
            pred_traj_4hz = pred_traj
        else:
            raise ValueError(
                f"pred_future点数不支持: sample_id={frame_name}, got N={num_points} "
                f"(期望5(1Hz)、10(2Hz)或20(4Hz))"
            )

        # 只在开头打印少量调试信息，避免刷屏
        if i < 3:
            print(f"[debug] sample_id={frame_name}, input_points={num_points}, pred_traj_4hz.shape={pred_traj_4hz.shape}")

        # 创建TrajectoryPrediction
        trajectory_prediction = wod_e2ed_submission_pb2.TrajectoryPrediction(
            pos_x=pred_traj_4hz[:, 0].astype(np.float32).tolist(),
            pos_y=pred_traj_4hz[:, 1].astype(np.float32).tolist()
        )

        # 创建FrameTrajectoryPredictions
        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
            frame_name=frame_name,
            trajectory=trajectory_prediction
        )

        frame_predictions.append(frame_trajectory)

    print(f"完成轨迹转换，共 {len(frame_predictions)} 个预测")

    # 分片并保存
    sub_file_names = [
        os.path.join(output_dir, f'mysubmission.binproto-{i:05d}-of-{num_shards:05d}')
        for i in range(num_shards)
    ]

    num_predictions_per_shard = math.ceil(len(frame_predictions) / num_shards)
    print(f"每个分片包含约 {num_predictions_per_shard} 个预测")

    submissions = []
    for i in range(num_shards):
        start = i * num_predictions_per_shard
        end = min((i + 1) * num_predictions_per_shard, len(frame_predictions))

        submission = wod_e2ed_submission_pb2.E2EDChallengeSubmission(
            predictions=frame_predictions[start:end]
        )

        # 设置提交信息
        submission.submission_type = wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        submission.authors[:] = authors
        submission.affiliation = affiliation
        submission.account_name = account_name
        submission.unique_method_name = method_name
        submission.method_link = method_link
        submission.description = description
        submission.uses_public_model_pretraining = uses_public_model_pretraining
        submission.public_model_names.extend(public_model_names)
        submission.num_model_parameters = num_model_parameters

        # 保存分片
        with open(sub_file_names[i], 'wb') as fp:
            fp.write(submission.SerializeToString())

        print(f"保存分片 {i}: {sub_file_names[i]} (包含 {end - start} 个预测)")
        submissions.append(submission)

    print(f"\n转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"请运行以下命令打包提交文件:")
    print(f"  cd {os.path.dirname(output_dir)}")
    print(f"  tar cvf {os.path.basename(output_dir)}.tar {os.path.basename(output_dir)}")
    print(f"  gzip {os.path.basename(output_dir)}.tar")
    print(f"然后上传 {os.path.basename(output_dir)}.tar.gz 到挑战网站")

    return submissions


if __name__ == "__main__":
    # 配置参数
    ROOT_PATH = ""
    INPUT_JSON = f"{ROOT_PATH}/test_outputs.jsonl"
    OUTPUT_DIR = f"{ROOT_PATH}/waymo_submission"


    # 提交信息 - 请根据你的实际情况修改
    submissions = convert_predictions_to_submission(
        input_json_path=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        num_shards=10,  # 可以根据数据量调整
        authors=['Anonymous'],  # 修改为你的姓名
        affiliation='Anonymous',  # 修改为你的机构
        account_name='',  # 修改为你的邮箱
        method_name='Anonymous',  # 修改为你的方法名
        # method_link='https://github.com/yourrepo ',  # 修改为你的方法链接
        description='',  # 修改为你的方法描述
        uses_public_model_pretraining=True,  # 是否使用预训练模型
        public_model_names=['Qwen3.5-2B'],  # 如果使用预训练模型，列出模型名称
        num_model_parameters='2B'  # 模型参数量
    )

    print("\n转换成功！")
