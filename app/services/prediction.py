from app.model import ScamDetectionModel
from app.preprocessing import PreprocessedInput, preprocess_job_post
from app.schemas import JobPostInput, PredictResponse


def predict_single(job_post: JobPostInput, model: ScamDetectionModel) -> PredictResponse:
    preprocessed = preprocess_job_post(
        job_title=job_post.job_title,
        job_desc=job_post.job_desc,
        company_profile=job_post.company_profile,
        skills_desc=job_post.skills_desc,
        salary_range=job_post.salary_range,
        employment_type=job_post.employment_type,
    )
    result = model.predict(preprocessed.combined_text)
    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        scam_probability=result["scam_probability"],
        warning_signals=preprocessed.warning_signals,
    )


def predict_batch(
    job_posts: list[JobPostInput], model: ScamDetectionModel
) -> list[PredictResponse]:
    preprocessed_list: list[PreprocessedInput] = []
    for post in job_posts:
        preprocessed_list.append(
            preprocess_job_post(
                job_title=post.job_title,
                job_desc=post.job_desc,
                company_profile=post.company_profile,
                skills_desc=post.skills_desc,
                salary_range=post.salary_range,
                employment_type=post.employment_type,
            )
        )

    texts = [p.combined_text for p in preprocessed_list]
    results = model.predict_batch(texts)

    return [
        PredictResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            scam_probability=result["scam_probability"],
            warning_signals=preprocessed.warning_signals,
        )
        for result, preprocessed in zip(results, preprocessed_list)
    ]
