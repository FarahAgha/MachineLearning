import argparse


def get_args():
    """
    Get argument parser for train cli.
    Command line argument example:

    USAGE: python ./predict.py /path/to/image.jpg checkpoint.pth --top_k 5

    Returns an argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="Image prediction.",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth --top_k 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

   
    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return top KK most likely classes.',
                        )

    
    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'USAGE: python ./predict.py /path/to/image.jpg checkpoint.pth  --top_k 5')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
