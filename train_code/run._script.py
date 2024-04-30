import subprocess

# 명령어 실행
script = ['resnet50','resnet101','resnet152',"freeze_resnet50","freeze_resnet101","freeze_resnet152",'vgg11''vgg16','vgg19','freeze_vgg11','freeze_vgg16','freeze_vgg19'] 
for i in script:

    result = subprocess.run(['python3', 'double_wandb_sweep2.py', "--arch", str(i)], capture_output=True, text=True)

# 실행 결과 출력
print(result.stdout)
print('표준 에러:', result.stderr)
print('리턴 코드:', result.returncode)
