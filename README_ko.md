# Q1 2026 생성형 AI 확산 분석 재현 패키지

이 저장소는 논문 **Conversion Profiles in the Global Diffusion of Generative AI: Governance Readiness, Access, and Platform Observability**의 국가 단위 분석을 재현하기 위한 데이터와 코드 패키지입니다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/00_run_all.py
```

실행하면 Q1 2026 통합 데이터, 본문 표, 보충자료 표, 그림, 요약 보고서가 재생성됩니다.

## 주의 사항

- 세 가지 profile은 인과적 regime 분류가 아니라 진단적 profile입니다.
- Claude 지표는 Claude-visible platform use로 해석해야 하며, 전체 specialist adoption을 포괄하는 직접 측정치가 아닙니다.
- access-adjusted residual은 welfare나 productivity가 아니라 broad diffusion의 진단적 잔차입니다.
- double-anonymized review에서는 저자 식별 정보가 드러나지 않는 방식으로 저장소 링크를 제공하는 것이 좋습니다.
