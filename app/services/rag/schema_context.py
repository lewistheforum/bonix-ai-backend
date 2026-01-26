
"""
Database schema context for the RAG chatbot.
This file contains the structural information about the database tables
relevant to doctor, clinic, and scheduling information.
"""

SCHEMA_DESCRIPTION = """
SYSTEM DATABASE SCHEMA:

The chatbot has access to the following database structure to understand the clinic system:

1. **doctor_information** (Stores details about doctors)
    - `account_id` (uuid): Links to the main account table.
    - `full_name` (text): Doctor's full name.
    - `gender` (enum): Male, Female, Other.
    - `academic_degree` (text): e.g., PhD, Master.
    - `experience` (text): Years of experience or description.
    - `position` (text): Current position.
    - `introduction_1` (jsonb): General introduction or biography.
    - `work_process_2` (jsonb): History of work experience and employment.
    - `study_process_3` (jsonb): History of education and degrees.
    - `members_4` (jsonb): Professional memberships and affiliations.
    - `scientific_work_5` (jsonb): Published scientific works and research.
    - `papers_6` (jsonb): Awards, recognitions, or additional papers.
    - `dob` (date): Date of birth.

2. **clinic_admin_information** (Stores details about clinic administrators - main clinic level)
    - `account_id` (uuid): Links to the main account table.
    - `clinic_name` (text): Name of the main clinic.
    - `description` (text): General description of the clinic.
    - `specialized_in` (jsonb): Specializations offered by the clinic.
    - `pros` (jsonb): Key advantages or strengths of the clinic.
    - `paraclinical` (jsonb): Paraclinical services provided.
    - `dob` (date): Date of birth of the administrator.

3. **clinic_manager_information** (Stores details about clinic managers - branch level)
    - `account_id` (uuid): Links to the main account table.
    - `clinic_branch_name` (text): Name of the specific clinic branch.
    - `full_name` (text): Full name of the manager.
    - `gender` (enum): Male, Female, Other.
    - `dob` (date): Date of birth.

4. **clinic_staff_information** (Stores details about clinic staff)
    - `account_id` (uuid): Links to the main account table.
    - `full_name` (text): Staff's full name.
    - `gender` (enum): Male, Female, Other.
    - `clinic_role` (text): Role within the clinic.
    - `dob` (date): Date of birth.

5. **addresses** (Stores address details for clinics or users)
    - `account_id` (uuid): Links to the main account table.
    - `address` (text): Specific street address.
    - `ward` (number): Administrative ward code.
    - `district` (number): Administrative district code.
    - `province` (number): Administrative province code.
    - `province_name` (text): Name of the province.
    - `district_name` (text): Name of the district.
    - `ward_name` (text): Name of the ward.

6. **feedbacks** (Patient reviews and ratings)
    - `appointment_id` (uuid): Links to the specific appointment being reviewed.
    - `clinic_id` (uuid): Links to the clinic being reviewed.
    - `doctor_id` (uuid, optional): Links to the specific doctor being reviewed.
    - `rating` (smallint): 1-5 star rating given by the patient.
    - `description` (text): Text content of the review or feedback.
    - `type` (enum): Type of feedback (CLINIC or DOCTOR).

7. **clinic_service_category** (Categories of medical services)
    - `category_name` (text): Official name of the service category.
    - `type` (enum): Type or classification code for the category.
    - `is_active` (boolean): Whether the category is currently active.

8. **clinic_services** (Specific medical services offered)
    - `category_id` (text): Links to the service category.
    - `service_name` (text): Name of the specific service.
    - `service_code` (text): Unique identifier code for the service.
    - `description` (text): Detailed description of what the service entails.
    - `service_functions` (text[]): List of functions or purposes associated with the service.
    - `is_active` (boolean): Whether the service is currently active.

9. **clinic_service_config** (Configuration of services for specific clinics)
    - `service_id` (uuid): Links to the general clinic service.
    - `clinic_id` (uuid): Links to the clinic offering the service.
    - `price` (numeric): Cost of the service at this clinic.
    - `discount` (numeric): Discount percentage applied to the service price.
    - `duration_min` (integer): Estimated duration of the service in minutes.
    - `note_for_patient` (text): Specific notes or instructions for the patient.
    - `is_active` (boolean): Whether the service is currently active at this clinic.

10. **employee_schedule** (Doctor and employee work schedules)
    - `employee_id` (uuid): Links to the doctor or employee.
    - `clinic_id` (uuid): Links to the clinic where they are working.
    - `clinic_shift_id` (uuid): Links to the defined clinical shift.
    - `work_date` (date): Specific calendar date of the work schedule.
    - `week_day` (enum): Day of the week (MONDAY, TUESDAY, etc.).

11. **clinic_shift** (Definitions of work shifts)
    - `shift` (enum): Time period of the shift (morning, afternoon, evening).

12. **clinic_shift_hour** (Specific hours for defined shifts)
    - `shift_id` (uuid): Links to the parent shift definition.
    - `start_hour` (time): Start time of the shift slot (e.g., 08:00).
    - `end_hour` (time): End time of the shift slot (e.g., 12:00).
    - `limit` (smallint): Maximum number of appointments allowed in this slot.

13. **clinic_room_employee_schedule** (Mapping of employees to rooms)
    - `clinic_room_id` (uuid): Links to the physical clinic room.
    - `employee_schedule_id` (uuid): Links to the employee's scheduled shift.

14. **clinic_room** (Physical rooms within a clinic)
    - `clinic_id` (uuid): Links to the clinic.
    - `room_name` (text): Name or number of the room.

15. **appointments** (Records of patient appointments)
    - `clinic_id` (uuid): Links to the clinic.
    - `patient_id` (uuid): Links to the patient.
    - `room_name` (text): Name of the room assigned for the appointment.
    - `doctor_shift_hour_id` (uuid): Links to the specific time slot for the doctor.
    - `doctor_id` (uuid): Links to the doctor.
    - `extra_hour` (time): Any extra time allocated for the appointment.
    - `appointment_date` (date): Date of the appointment.
    - `appointment_hour` (time): Specific time of the appointment.
    - `total` (numeric): Total cost of the appointment.
    - `status` (enum): Current status (PENDING, CONFIRMED, CANCELLED, etc.).
    - `is_reminder` (boolean): Flag to indicate if a reminder should be sent.
    - `patient_note` (text): Notes provided by the patient.
    - `reject_reason` (text): Reason for rejection if the appointment was rejected.

16. **blogs** (Blog posts and informational content)
    - `_id` (uuid): Unique identifier for the blog.
    - `clinic_id` (uuid): Links to the clinic that created the blog.
    - `title` (text): Title of the blog post.
    - `content` (text): Full content of the blog post.
    - `thumbnail` (text): URL or path to the thumbnail image.
    - `type` (text): Type or category of the blog post.

17. **general_accounts** (Patient user information)
    - `account_id` (uuid): Links to the main account table.
    - `full_name` (text): Patient's full name.
    - `gender` (enum): Male, Female, Other.
    - `dob` (date): Date of birth.

18. **accounts** (Main account table - users of the system)
    - `_id` (uuid): Unique identifier.
    - `username` (text): Login username.
    - `email` (text): Email address.
    - `phone` (text): Phone number.
    - `role` (enum): User role (PATIENT, DOCTOR, CLINIC_ADMIN, CLINIC_MANAGER, CLINIC_STAFF).
    - `status` (text): Account status.
    - `parent_id` (uuid): Links to parent account (for hierarchy).
    - `ban_counts` (integer): Number of times the account has been banned.
    - `ban_description` (text): Description of ban reason if applicable.
"""

