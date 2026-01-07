## NovaSR: Pushing the Limits of Extreme Efficiency in Audio Super-Resolution
This is the repository for NovaSR, a tiny 50kb audio upsampling model that upscales muffled 16khz audio into clear and crisp 48khz audio at speeds over 3500x realtime!

### Key benefits
* Speed: Can reach 3600x realtime speed on a single gpu.
* Quality: On par with models 5,000x larger.
* Size: Just 52kb in size, several thousand times smaller then most.

### Why is this even useful?
* Enhancing models: Most TTS and audio models produce muffled 16khz/24khz audio so NovaSR can enhance the clarity and crispness with nearly 0 computation cost.
* Real-time enhancement: NovaSR allows for on device enhancement of any low quality audio while using nearly no memory.
* Restoring datasets: Many datasets are poor quality and NovaSR can considerably enhance their clarity and crispness within seconds instead of hours.

| Model         | Speed (Real-Time) | Model Size |
| :------------ | :---------------- | :--------- |
| **NoraSR** | **3600x Faster** | **~0.05MB** |
| AP-BWE        | 200x Faster       | ~100 MB      |
| FlowHigh      | 20x Faster        | ~450 MB     |
| FlashSR       | 14x Faster        | ~1000 MB     |
| AudioSR       | 0.6x (Slower)     | ~2000 MB     |


