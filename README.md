# RNN을 이용한 텍스트 생성기(Text Generator using RNN)
샘플 코드를 기반으로 한 텍스트 생성 프로그램입니다.(Text Generator based on sample code)
### 한국어 설명
그럴듯한 텍스트 생성을 위해서는 최소 5000 Epoch의 학습이 필요합니다.
<br> 이 코드는 Keras 예제, Tensorflow 예제와 같은 코드를 참고하여 제작되었습니다.
<br>
<br> 몇가지 유의사항:
<br>- 이 프로그램은 대용량 텍스트 처리에 적합하지 <b>않습니다</b>.
<br>- 이 프로그램은 모든 언어에 대응합니다. (일단, 한국어 구동은 확인했습니다.)
<br>- RNN의 계산량이 매우 많음에 따라 GPU환경에서 구동하는 것을 추천드립니다.
<br>
### EN Version.
At least 5000 epochs are required before the generated text starts sounding coherent.
<br> *(This model is inspired by many examples, such as Keras samples, or tensorflow tutorials.)*
<br>
<br> Few notices:
<br>- This script is <b>NOT</b> designed for large text.
<br>- This script is designed for ANY language. (It works for my language, at least.)
<br>- It is recommended to run this script on GPU, as recurrent networks are quite computationally intensive.
