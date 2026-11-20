# Afghanistan Health Response: 4W Monitoring and Data Quality

Interactive dashboard of Afghanistan Health Cluster partner 4W reports, Jan 2022 – Jul 2023 (72,820 facility-month-indicator rows, 34 provinces).

**Live:** https://danyalkhattak-git.github.io/afghanistan-health-dashboard/

- **Response:** services delivered by month, province (map), sex and age group, partner and facility type, for 22 indicators.
- **Data quality:** partner reporting completeness and timeliness, duplicate rows, values far above a facility's usual level, and indicators renamed mid-period.

Figures count services, not unique people, and are never summed across indicators. `prep/who_prep.py` contains the cleaning and quality checks; the raw data is not included.

Danyal Khattak
