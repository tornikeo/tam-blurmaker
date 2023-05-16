## Install and run

### Pre-requisites
- [Docker](https://docs.docker.com/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Nvidia docker Containers

### Build and run

default arguments (input video is `test_sample/mall_480.mp4`, default tracking args `test_sample/mall_480/sample_track_person.json`):

```bash
./run.sh
```

custom arguments:

<!-- parser.add_argument('--input', type=Path, default=Path('test_sample/mall_480.mp4'))
parser.add_argument('--track_data', type=Path, default=Path('test_sample/mall_480/sample_track_person.json'))
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--sam_model_type", type=str, default="vit_b")
parser.add_argument('--output_video', type=Path, default='result.mp4')
parser.add_argument('--output', type=Path, default='output.json')
parser.add_argument("--debug", type=bool, default=True) -->

```bash
./run.sh --input <input_video> \
      --track_data <track_data> \
      --device <device> \
      --sam_model_type <sam_model_type> \
      --output_video <output_video> \
      --output <output> \
      --debug <debug>
```

