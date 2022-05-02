Lembrar de tornar o arquivo run_app executavel.

Para buildar a imagem: docker build -t toy_deploy .


Para executar o container: docker container run --rm -p 8000:8000 -it toy_deploy

Url para predict: http://172.17.0.2:8080/predict?x1=1&x2=10