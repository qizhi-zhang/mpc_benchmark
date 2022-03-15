
VERSION = v0.97
rosetta:
	docker build . -t rosetta:$(VERSION)
	docker tag rosetta:$(VERSION) registry.cn-hangzhou.aliyuncs.com/dtunion/rosetta:$(VERSION)
	echo "build rosetta docker image ok"
	docker push registry.cn-hangzhou.aliyuncs.com/dtunion/rosetta:$(VERSION)