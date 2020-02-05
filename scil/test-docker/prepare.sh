#!/bin/bash

cd $(dirname "$0")

echo "Checking docker"
docker ps
if [ $? != 0 ] ; then
	echo "Error, cannot run docker commands"
	groups|grep docker || echo "You are not in the docker group !"
	exit 1
fi

echo "Building docker containers"

docker build -t kunkel/scil:ubuntu14.04 ubuntu14.04
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi
docker build -t kunkel/scil:ubuntu16.04 ubuntu16.04
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi

docker build -t kunkel/scil:ubuntu17.04 ubuntu17.04
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi

docker build -t kunkel/scil:centos6 centos6
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi
docker build -t kunkel/scil:centos7 centos7
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi

docker build -t kunkel/scil:archlinux archlinux
if [ $? != 0 ] ; then
	echo "Error building image"
	exit 1
fi
