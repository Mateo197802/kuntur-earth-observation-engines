# External evaluation protocol v1.0

**Status:** Frozen before external corpus inspection for KUNTUR model/analysis results.  
**Permitted use:** External station records are evaluation-only. They must not enter forecast fitting, feature scaling, augmentation scale estimation, hyperparameter selection, model selection, or training-only climatology.

## Source selection

Priority temperature reference: NOAA/NCEI Global Historical Climatology Network Monthly (GHCN-M) v4, quality-controlled unadjusted mean monthly temperature (`TAVG`, qcu) and station inventory. Use final quality-controlled (`qcf`) only as a sensitivity check after the primary source and station rules are frozen. The GHCN-M release combines multiple source streams, including GHCN-Daily and ISTI; it is not described as independent merely because it is a separate product. NOAA documents `TAVG` stored as whole integers in hundredths of a degree Celsius; divide by 100 to obtain °C. Reject missing `-9999` sentinels and nonblank QC flags in the primary qcu series. Retain data-measurement and source flags (DMFLAG/DSFLAG) and original integer values; DMFLAG is not itself a QC-failure flag.

Protocol correction note: the official GHCN-M v4 readme was checked before any external data file was downloaded. This frozen unit/flag wording supersedes the initial draft; no engine performance or external observations were inspected.

Explore INAMHI public access first. If the relevant station archive requires credentials or project-specific authorization not already available, do not attempt access using personal data; record the exact access limitation and continue with public NOAA data. NOAA GHCN-Daily may be used for station-level metadata/QC sensitivity where monthly data are sparse, but daily and monthly products are not treated as independent observations.

For precipitation corroboration, investigate CHIRPS monthly. It is a satellite/station blended product and is a cross-product comparison, not independent ground truth. For vegetation, attempt only a bounded public Landsat Collection 2 sample with QA and cloud masking; record access/license limits. MODIS v1 strict-QA coverage is low, so vegetation comparisons require a matched spatial/time support and explicit valid-area report.

## Geography and eligibility

Use only stations with valid coordinates inside the frozen preliminary bbox `[-81.5, -5.1, -75.0, 1.7]`, and keep their individual identities. Deduplicate station IDs; flag colocated station IDs within 1 km and report station-network changes. Include only months with at least 3 eligible stations in the equal-station composite; report station count and spatial dispersion monthly. This station composite is a convenience diagnostic, not an area-weighted Ecuador mean. Coast/Andes/Amazon or external-region labels are prohibited without approved masks.

For temperature, compare the NOAA monthly mean and KUNTUR regional forecast/mean only as **observational corroboration with representativeness limitations**. A station point/network average is not the same estimand as a 0.5°×0.625°-and-larger area-weighted bbox mean. If a station-matched MERRA-2 cell is not locally available, do not claim collocated-grid validation. Report correlation, bias, MAE, RMSE, temporal overlap, station counts, and sensitivity to station inclusion only descriptively; do not use them for selection.

For precipitation, compare matched monthly precipitation totals only when units, calendar coverage, and QA pass checks; label CHIRPS-vs-IMERG as cross-product corroboration. For MODIS/Landsat, compare only overlap months and spatially aggregated valid pixels after QA; retain source-specific grids and do not interpolate to imply equal support.

## Freeze, blinding, and reporting

Record source URL/version/access date, license/terms, station inventory hash, selected station IDs, QC rules, overlap window, aggregation, and code hash before calculating comparison metrics. Do not choose stations after inspecting model residuals. Report missing periods, changing station counts, homogenization status, possible shared inputs, and domain/representativeness limits. External metrics are separate from internal validation, 38-month robustness, and the final 2022–2024 holdout. If a source cannot be obtained, mark it unavailable with evidence and continue independently.
