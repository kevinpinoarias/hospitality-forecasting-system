# School calendar: the dates and where they come from

The model's school-holiday flags (`is_school_holiday`, `is_christmas_break`, `is_easter_break`,
`is_summer_break`) come from `SCHOOL_BREAKS` in `src/features/build_features.py`. This page records every
date in it, the sources it was checked against and how sure we are. Checked 12 October 2026.

## The convention

A break runs from the **first day without pupils** after the last school day to the **last day before
pupils return**. The closing day itself (a school day, often a half day) is not in the break. The weekends
on either side are inside it, and so are staff in-service days next to a break, because pupils are on
holiday on those days. The original table used Monday-to-Friday spans only, so the first weekend of every
holiday was labelled term time.

## What could and could not be checked

- **Glasgow City Council's own term-dates page** (`glasgow.gov.uk/article/4276/School-Term-Dates`) sits behind a
  bot-check that this project does not try to bypass, so it could not be read automatically. On 12 October 2026 the
  project owner opened it and supplied screenshots of the **2026-27** session; every date shown matches the table
  (rows marked "council page" below). The page now shows only the current session, so 2023-24, 2024-25 and 2025-26
  rest on the independent sources.
- The council's own *School Letting Holidays* PDF (April 2026 to March 2027) lists the spring break as Friday
  3 to Sunday 19 April 2026, which is exactly the stored Easter 2026 span; it was read through a search snippet only.
- Each date below was checked against at least two independent sources: the Glasgow World term-time
  articles (2023-24, 2024-25 and 2025), term-date pages and PDFs published by Glasgow schools
  (Jordanhill's PDFs for 2023-24, 2024-25 and 2026-27), and two aggregator pages that quote the council
  (`ukschoolholidays.co.uk`, `eduscot.co.uk`). A search of the council's site also returned its 2026-27 and
  2027-28 pupil-return dates, which match.
- **Jordanhill sets its own dates** (its Board of Managers) and often closes a day earlier than the council
  (for example summer 2024: Tuesday 25 June against the council's Wednesday 26 June), so it is used to
  confirm return dates and the shape of each break, not the closing day.

## The table

Last school day, then the pupil-free span stored in `SCHOOL_BREAKS` (inclusive), then the day pupils return.

| Break | Last school day | Stored span | Pupils return | Confidence |
|---|---|---|---|---|
| Christmas 2023-24 | Fri 22 Dec 2023 | 23 Dec 2023 to 7 Jan 2024 | Mon 8 Jan 2024 | High: two sources |
| Easter 2024 | Thu 28 Mar 2024 | 29 Mar to 14 Apr 2024 | Mon 15 Apr 2024 | High: two sources |
| Summer 2024 | Wed 26 Jun 2024, 1 pm | 27 Jun to 13 Aug 2024 | Wed 14 Aug 2024 | Return high; closing day one council source (Jordanhill: Tue 25 Jun) |
| Christmas 2024-25 | Fri 20 Dec 2024 | 21 Dec 2024 to 5 Jan 2025 | Mon 6 Jan 2025 | High: two sources |
| Easter 2025 | Fri 4 Apr 2025, 2.30 pm | 5 to 21 Apr 2025 | Tue 22 Apr 2025 | High: three sources |
| Summer 2025 | Wed 25 Jun 2025, 1 pm | 26 Jun to 13 Aug 2025 | Thu 14 Aug 2025 | Return high; closing day two sources (Jordanhill: Tue 24 Jun) |
| Christmas 2025-26 | Fri 19 Dec 2025, 2.30 pm | 20 Dec 2025 to 4 Jan 2026 | Mon 5 Jan 2026 | High: three sources |
| Easter 2026 | Thu 2 Apr 2026, 2.30 pm | 3 to 19 Apr 2026 | Mon 20 Apr 2026 | High: three sources and the council's letting-holidays PDF |
| Summer 2026 | Thu 25 Jun 2026, 1 pm | 26 Jun to 11 Aug 2026 | Wed 12 Aug 2026 | Return: council page. Closing day: two sources |
| Christmas 2026-27 | Fri 18 Dec 2026, 2.30 pm | 19 Dec 2026 to 4 Jan 2027 | Tue 5 Jan 2027 | **Council page** (and three sources) |
| Easter 2027 | Fri 2 Apr 2027, 2.30 pm | 3 to 18 Apr 2027 | Mon 19 Apr 2027 | **Council page** (and three sources) |
| Summer 2027 | Fri 25 Jun 2027, 1 pm | 26 Jun to 17 Aug 2027 | Wed 18 Aug 2027 | Closing day: **council page**. Return: council search snippet, Jordanhill, eduscot |

The 2027 rows change no training row; they let the API flag school holidays on the dates it forecasts.
Dates after the last row get no flag. 2027-28 holidays beyond summer 2027 are published by the council but one
aggregator marks some of them as estimated, so they are not added yet.

### October week and February mid-term

The same convention. The Friday in-service day before the October week, and the Wednesday in-service day after the
February mid-term (Monday and Tuesday), are pupil-free, so they are inside the span.

| Break | Last school day | Stored span | Pupils return | Confidence |
|---|---|---|---|---|
| October 2023 | Thu 12 Oct 2023 | 13 to 22 Oct 2023 | Mon 23 Oct 2023 | High: two sources (before the data starts) |
| February 2024 | Fri 9 Feb 2024 | 10 to 14 Feb 2024 | Thu 15 Feb 2024 | High: two sources |
| October 2024 | Thu 10 Oct 2024 | 11 to 20 Oct 2024 | Mon 21 Oct 2024 | High: two sources |
| February 2025 | Fri 14 Feb 2025 | 15 to 19 Feb 2025 | Thu 20 Feb 2025 | High: two sources |
| October 2025 | Thu 9 Oct 2025 | 10 to 19 Oct 2025 | Mon 20 Oct 2025 | High: three sources |
| February 2026 | Fri 13 Feb 2026 | 14 to 18 Feb 2026 | Thu 19 Feb 2026 | High: a handbook and Glasgow World; one aggregator lists a return on 23 Feb, treated as an error because every other source and every other year's pattern gives the Thursday |
| October 2026 | Thu 8 Oct 2026 | 9 to 18 Oct 2026 | Mon 19 Oct 2026 | **Council page** (in-service Fri 9, October week 12 to 16); return date from Jordanhill and eduscot |
| February 2027 | Fri 12 Feb 2027 | 13 to 17 Feb 2027 | Thu 18 Feb 2027 | **Council page** (Mon 15, Tue 16, Wed 17 in-service); return date from Jordanhill and eduscot |

**Not in the table:** the Good Friday weekend of 2027 (a separate four-day closure), single-day closures
(in-service days elsewhere in the year, the May Day holiday, the September and May long weekends) and the council's
Spring and Autumn bank-holiday weekends.

## What the original table got wrong

The original table (`SCHOOL_BREAKS_V1`, used by Series 1 to 28) started at Christmas 2024, so 2024's Easter and
summer holidays and the Christmas holiday that ends on 7 January 2024 were all treated as term time. Its dates
for the holidays it did cover:

| Break | Original | Verified | Difference |
|---|---|---|---|
| Christmas 2024-25 | 23 Dec to 3 Jan | 21 Dec to 5 Jan | the two weekends were term |
| Easter 2025 | 7 to 18 Apr | 5 to 21 Apr | Easter weekend and Easter Monday (19 to 21 Apr) were term |
| Summer 2025 | 30 Jun to 15 Aug | 26 Jun to 13 Aug | started 4 days late (including Saturday 28 Jun 2025, the busiest day in the data); ended 2 days late (14 and 15 Aug were school days) |
| Christmas 2025-26 | 22 Dec to 5 Jan | 20 Dec to 4 Jan | started 2 days late; 5 Jan was a school day |
| Easter 2026 | 30 Mar to 10 Apr | 3 to 19 Apr | wrong week: 30 Mar to 2 Apr were school days, 11 to 19 Apr were missing |
| Summer 2026 | 29 Jun to 14 Aug | 26 Jun to 11 Aug | started 3 days late; ended 3 days late |
| Christmas 2026-27 | 21 Dec to 4 Jan | 19 Dec to 4 Jan | the first weekend was term |
| Easter 2027 | 5 to 16 Apr | 3 to 18 Apr | the two weekends were term |
| Summer 2027 | 28 Jun to 10 Aug | 26 Jun to 17 Aug | ended a week early |

The effect on the model is measured in Series 29 (`docs/EXPERIMENT_LOG.md`). Series 1 to 28 used the original
table; to reproduce their figures exactly, build the features with `SCHOOL_BREAKS_V1`.

## Sources

- Glasgow World, [school term times 2023/24](https://www.glasgowworld.com/news/educationglasgow-schools-term-times-2023-24-4253766) and [2024/25](https://www.glasgowworld.com/news/glasgow-school-holidays-20242025-all-of-this-years-school-holiday-dates-in-glasgow-4689799), and the 2025 [bank and school holidays](https://www.glasgowworld.com/news/everything-you-need-to-know-about-glasgow-bank-holidays-and-school-holidays-for-2025-4933834) and [term dates](https://www.glasgowworld.com/education/scotland-school-holidays-2025-full-list-term-dates-breaks-february-half-term-easter-4917317) articles.
- Jordanhill School term dates: [2023-24](https://www.jordanhill.glasgow.sch.uk/wp-content/uploads/2023/11/Term-Dates-2023-24.pdf), [2024-25](https://www.jordanhill.glasgow.sch.uk/wp-content/uploads/2023/08/Term-Dates-2024-25.pdf) and [2026-27](https://www.jordanhill.glasgow.sch.uk/wp-content/uploads/2025/08/Term-Dates-2026-27-FINAL.pdf).
- [ukschoolholidays.co.uk, Glasgow](https://ukschoolholidays.co.uk/scotland-school-holidays/glasgow/) and [eduscot.co.uk, Glasgow City](https://eduscot.co.uk/school-holidays/glasgow-city).
- Glasgow City Council, [School Term Dates](https://www.glasgow.gov.uk/article/4276/School-Term-Dates): the authority; read only through search snippets (see above).
