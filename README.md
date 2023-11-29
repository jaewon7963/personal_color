# personal_color

실제 사용은 tone_analysis.py 와 tone_check.py 두 스크립트만 주로 사용하였습니다.

tone_analysis.py 는 사용자의 입력(이미지)을 받아서 톤을 진단해주는 스크립트이며,

tone_check.py 는 모델의 정확도를 판단하기 위하여, 저장해둔 이미지 여러장을 분석하는 스크립트 입니다.

다만, tone_check.py 에서는 이미지의 이마, 뺨, 턱 의 색상들이 필요합니다. 

그렇기에 eye_extract.py 에서 작성되어 있는 함수인
cheek_extracting()
chin_extracting()
forehead_extracting()
들을 이용해서, 해당 부위들을 먼저 추출합니다. 그 후, extracting_cheek_color.py 의 입력으로 추출한 부위들이 저장된 경로를 사용하면
해당 부위들을 대표하는 색상을 출력값으로 얻을 수 있습니다.

tone_check.py 에서는 이름만 입력받으면, 뺨, 턱, 이마의 색상이 저장된 경로를 찾게 설정을 해놨습니다.
그러니, 각각의 부위를 ./YOUR/PATH/cheek/NAME , ./YOUR/PATH/chin/NAME , ./YOUR/PATH/forehead/NAME 과 같은 형식으로 저장하는 걸 추천합니다 !

![image](https://github.com/jaewon7963/personal_color/assets/81609477/1c960bdd-9d61-4c9d-87f7-039f2912d5b6)
위의 이미지가 tone_check.py를 실행한 결과 입니다.
gt값을 설정하고, 입력경로를 설정해주면 해당 경로에 저장된 이미지들이 gt로 설정한 값과 일치한지 확인합니다.
