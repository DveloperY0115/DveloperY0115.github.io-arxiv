---
layout: post
title: "애플 M1 심층분석"
background: '/assets/post-images/AppleSoC/0.jpg'
---

# 시작하며

![신형 MacBook Pro 소개 영상](https://www.youtube.com/watch?v=9tobL8U7dQo)

한국 시간 2021년 10월 19일 오전 2시 (현지 시각 오전 10시), 마침내 베일에 싸여 있던 애플의 차세대 SoC M1 Pro와 M1 Max가 공개되었습니다. 이와 더불어 지난 2016년 이후 디자인 변화 없이 마이너 업데이트를 거쳐온 MacBook Pro 제품군에도 변화가 찾아왔는데요. '봉인 해제 (Unleashed)'라는 이벤트의 제목에 걸맞게 구속에서 벗어나 날뛰는 야수같은 성능을 보여주며 기대감을 한껏 높였습니다. 강력해진 성능과 별개로 5년 넘게 논쟁의 대상이었던 Touch Bar가 물리 기능 키로 대체되고 마찬가지로 이전 세대 리뉴얼과 함께 자취를 감추었던 MagSafe와 여러 포트들이 되돌아오는 등 소소하지만(?) 반가운 변화들도 눈에 띄었습니다.

애플이 프로 유저들의 피드백을 수용하고 그들의 니즈를 충족시켜 줄 제품을 출시했다는 데엔 이견이 없고 이전 세대 MacBook Pro와 비교하여 개선된 부분이 한둘이 아니라는 점에도 동의합니다만, 오늘 포스트에서 중점적으로 다룰 것은 MacBook Pro라는 제품이 아닙니다. 본 글에서는 MacBook Pro와 더불어 MacBook Air, Mac Mini, 그리고 비교적 최근에 리뉴얼 된 iMac의 심장이자 두뇌인 애플의 자체 SoC, M1 칩셋군 (family of chips)에 집중합니다. iPhone과 iPad에 탑재된 A 시리즈 칩셋을 개발한 기간을 합치더라도 15년이 채 되지 않는 역사를 가진 '신입' 애플이 빚어낸 칩이 오랜 기간 CPU, GPU를 개발해온 인텔, AMD의 것과 대등한, 혹은 그 이상의 성능을 낼 수 있는 이유는 무엇일까요? 야수같은 성능 뒤에는 어떤 비밀이 감추어져 있을까요?

# 서막: WWDC 2020에서 자체 개발 아키텍처로의 전환을 발표하다

**전성비 (전력 대비 성능 비)**라는 단어만큼 현상황을 잘 요약하는 단어가 없을 것 같습니다. 애플이 오랜 시간 자사에 CPU와 GPU를 공급해온 인텔, AMD와의 인연을 마무리 짓고 자신들이 직접 개발한 칩셋으로의 이전을 발표한 데에는 여러 가지 이유가 있겠지만, 그들이 표면적으로 강조한 요인은:

> "우리가 개발한 칩셋이 성능은 비슷한데 전기는 더 적게 소모한다"

였습니다. 이는 처음있는 일이 아닌데요, WWDC 2020으로부터 약 15년을 거슬러 올라간 WWDC 2005에서 당시 애플 CEO 스티브 잡스는 PowerPC에게 작별을 고하고 인텔 x86 아키텍처로의 전환 소식을 전했습니다. 당시에도 그가 꺼낸 카드는 전성비였습니다. PowerPC와 x86를 나란히 놓고 비교하며 전성비가 상대적으로 뒤떨어지는 PowerPC 대신 향후 출시될 Mac과 OS X (현 macOS)은 인텔의 x86를 기반으로 동작할 것이라고 예고했죠. 그리고 이 청사진은 이후 약 2년에 걸쳐 현실화 되었습니다. 오늘날 인텔, AMD로 대표되는 AMD64 (x86-64)와 ARM Holdings의 ARM이 지배하고 있는 시장을 보면 이때의 결정은 분명 옳았습니다.

<center>
    <img class="img-fluid" src="/assets/post-images/AppleSoC/1.png">
</center>
<span class="caption text-muted">2006년 애플 맥월드에서 前 애플 CEO 스티브 잡스가 前 인텔 CEO 폴 오텔리니에게 웨이퍼를 건내 받는 모습</span>

하지만 얄궂게도 2005년 당시 PowerPC를 상대로 승리를 거머쥐었던 인텔은 최근 몇 년간의 부진한 행적으로 인해 애플의 시험대 위에 오르게 됩니다. 특히, 가뜩이나 작아진 폼팩터 때문에 쓰로틀링이 빈번히 발생하여 프로 노트북이라는 브랜딩이 무색해진 MacBook Pro 사용자들의 불만이 컸습니다. 제한된 보드 크기와 쿨링 시스템 때문에 별도의 그래픽 카드 없이 인텔의 내장 GPU가 강제된 13인치 MacBook Pro의 경우 문제가 더 심각했습니다. 애플의 입장에선 현재의 문제를 타개할 훌륭한 대체재로서가 아니라, 문제를 일으키고 있는 골칫거리로 입지가 180도 달라진 것이죠. 결국 애플은 앞으로 자신들의 시스템에 인텔이 공급하는 칩을 사용하지 않기로 결단을 내립니다.

<center>
    <img class="img-fluid" src="/assets/post-images/AppleSoC/2.png">
</center>
<span class="caption text-muted">WWDC 2020에서 애플 CEO 팀 쿡이 Apple Silicon으로의 전환을 발표하는 모습</span>

이 발표에서 애플은 다가올 가을 (2020년 가을)에 공개될 칩이 더 나은 CPU 및 GPU 성능, A 시리즈 칩셋에 탑재되어 ML 태스크를 가속하고 있는 뉴럴 엔진 (Neural Engine)을 비롯하여 이미지 처리에 특화 된 이미지 신호 프로세서 (Image Signal Processor, ISP), 사용자 데이터 보호를 담당할 Secure Enclave 등 맞춤형 기술들을 품고 Mac을 한 단계 더 진보시킬 것이라고 말했습니다. 가장 중요하게도, 자신들이 설계한 칩이 이 모든 것을 더 적은 전력을 소모하면서도 충분히 달성할 수 있다는 것을 강조했습니다.

<center>
    <img class="img-fluid" src="/assets/post-images/AppleSoC/3.jpeg">
</center>
<span class="caption text-muted">애플이 밝힌 Apple Silicon의 여러 기능 및 특징점</span>

# Apple M1, 모습을 드러내다

