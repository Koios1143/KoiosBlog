# Koios Blog

> This repository is still under development!

## Introduction

There's a simple blog create by hexo before, but unfortunately the source code is not on github, and somehow I lost my source code.

So, to create a new blog, and also try to use some tecnic rhat I haven't tried before is the target of this project.

## Target

Currently, I decide to build with Vue.js, and should be deploy on github page with CI/CD.

But this is still a blog, which is a website mainly contain simple text content pages, so I would like to write articles with Markdown, so that will be easier.

## Contents

After the base of the blog been developed well, I'll start putting some articles on it, and by the hope that the following topic could be covered in this year.

- Self Introduction
- NTHU Lecture Notes
- Machine Learning Basic Note
- Deep Learning Note
- Reinforcement Leanring Note
- Paper Read Note
- About Japan Exchange Plan

## Current items

- Dockerfile
    Can build a base blog environment, but without some configuration of `vue-markdown-loader`
- BlogBase.tar
    A docker image file contain the full base blog environment

## Environment Details

| Items | version |
| ----- | ------- |
| Nodejs | `v20.10.0` |
| npm | `10.2.3` |
| vue-markdown-loader | `2.5.0` |
| vue-loader | `17.4.2` |
| vue-template-compiler | `2.7.16` |

## Run Instructions

```bash
docker load --input BlogBase.tar
docker run -p 8080:8080 -it blog:v1 /bin/bash
```