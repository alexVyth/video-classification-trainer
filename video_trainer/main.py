from video_trainer.loading.dataset import FstDataset


def main() -> None:
    dataset = FstDataset()
    video, label = dataset[0]
    print(video.shape)
    print(label)


if __name__ == '__main__':
    main()
