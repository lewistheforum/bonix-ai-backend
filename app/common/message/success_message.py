class SuccessMessage:
    INDEX = "SUCCESS"
    SERVER = "Server is running"
    
    # User Management
    USER_FETCH_SUCCESS = "Users fetched successfully"
    USER_CREATE_SUCCESS = "User created successfully"
    USER_UPDATE_SUCCESS = "User updated successfully"
    USER_DELETE_SUCCESS = "User deleted successfully"
    USER_BANNED_SUCCESS = "User banned successfully"
    USER_UNBANNED_SUCCESS = "User unbanned successfully"
    USER_RESTORED_SUCCESS = "User restored successfully"
    
    # Authentication
    LOGIN_SUCCESS = "User logged in successfully"
    GOOGLE_LOGIN_SUCCESS = "Google login successful"
    PAYMENT_CREATE_SUCCESS = "Payment QR created successfully"
    PAYMENT_UPDATE_SUCCESS = "Payment status updated successfully"

    # 2-Step Registration
    ACCOUNT_BASIC_CREATED = "Account created successfully. Please complete your profile in Step 2."
    ACCOUNT_PROFILE_COMPLETED = "Profile created successfully. Please request verification code via POST /mailer/send-verification-code."
    
    # Clinic Manager
    CLINIC_MANAGER_CREATED = "Clinic manager account created successfully"
    CLINIC_STAFF_CREATED_SUCCESS = "Clinic staff account created successfully with PENDING status. Staff must complete profile."
    CLINIC_DOCTOR_CREATED_SUCCESS = "Doctor account created successfully with PENDING status. Doctor must complete profile."
