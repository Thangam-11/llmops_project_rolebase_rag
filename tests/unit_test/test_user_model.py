# tests/unit/test_user_model.py
import pytest

from models.model import (
    User,
    Department,
    UserRole,
    DEPARTMENT_ACCESS,
)


def make_user(department: Department, role: UserRole) -> User:
    """
    Construct a User in memory only — never added to a session,
    never committed. Pure Python property testing, no DB needed.
    """
    return User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed",
        department=department,
        role=role,
    )


# ---------------------------------------------------------------------------
# 1. DEPARTMENT_ACCESS mapping integrity
# ---------------------------------------------------------------------------

def test_every_department_enum_value_has_an_access_entry():
    """
    Catches the case where a new Department is added to the enum
    but someone forgets to wire up its collection access — that
    person would silently fall through to the ['general'] default.
    """
    for dept in Department:
        assert dept in DEPARTMENT_ACCESS, f"{dept} missing from DEPARTMENT_ACCESS"


@pytest.mark.parametrize("department,expected_collections", [
    (Department.engineering, ["engineering", "general"]),
    (Department.hr,          ["hr", "general"]),
    (Department.finance,     ["finance", "general"]),
    (Department.marketing,   ["marketing", "general"]),
    (Department.general,     ["general"]),
    (Department.c_level,     ["engineering", "hr", "finance", "marketing", "general"]),
])
def test_department_access_mapping_is_correct(department, expected_collections):
    assert DEPARTMENT_ACCESS[department] == expected_collections


@pytest.mark.parametrize("department", [
    Department.engineering, Department.hr, Department.finance, Department.marketing,
])
def test_non_general_departments_always_include_general(department):
    """
    'general' info should be visible to every department — confirms
    no department is accidentally locked out of shared/general content.
    """
    assert "general" in DEPARTMENT_ACCESS[department]


def test_no_department_can_see_departments_it_should_not():
    """
    Explicit negative checks — confirms cross-department leakage is impossible
    at the mapping level for every non-privileged department.
    """
    assert "hr" not in DEPARTMENT_ACCESS[Department.engineering]
    assert "finance" not in DEPARTMENT_ACCESS[Department.engineering]
    assert "marketing" not in DEPARTMENT_ACCESS[Department.engineering]

    assert "engineering" not in DEPARTMENT_ACCESS[Department.hr]
    assert "finance" not in DEPARTMENT_ACCESS[Department.hr]

    assert "engineering" not in DEPARTMENT_ACCESS[Department.finance]
    assert "hr" not in DEPARTMENT_ACCESS[Department.finance]

    assert "engineering" not in DEPARTMENT_ACCESS[Department.general]
    assert "hr" not in DEPARTMENT_ACCESS[Department.general]
    assert "finance" not in DEPARTMENT_ACCESS[Department.general]
    assert "marketing" not in DEPARTMENT_ACCESS[Department.general]


def test_c_level_is_the_only_department_with_full_access():
    all_collections = {"engineering", "hr", "finance", "marketing", "general"}
    for dept, collections in DEPARTMENT_ACCESS.items():
        if dept == Department.c_level:
            assert set(collections) == all_collections
        else:
            assert set(collections) != all_collections, (
                f"{dept} unexpectedly has full cross-department access"
            )


# ---------------------------------------------------------------------------
# 2. allowed_collections property — the actual enforcement point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("department,role,expected", [
    (Department.engineering, UserRole.viewer,  ["engineering", "general"]),
    (Department.hr,          UserRole.analyst, ["hr", "general"]),
    (Department.finance,     UserRole.manager, ["finance", "general"]),
    (Department.marketing,   UserRole.viewer,  ["marketing", "general"]),
    (Department.general,     UserRole.viewer,  ["general"]),
    (Department.c_level,     UserRole.viewer,  ["engineering", "hr", "finance", "marketing", "general"]),
])
def test_allowed_collections_matches_department_for_non_admin_roles(department, role, expected):
    user = make_user(department, role)
    assert user.allowed_collections == expected


@pytest.mark.parametrize("department", list(Department))
def test_admin_role_always_gets_full_access_regardless_of_department(department):
    """
    Critical override test: even an admin whose `department` field is
    'general' or 'marketing' must still get full cross-department access
    because of role, not department. If someone refactors this property
    and the admin check gets dropped or reordered, this test catches it.
    """
    user = make_user(department, UserRole.admin)
    assert set(user.allowed_collections) == {
        "engineering", "hr", "finance", "marketing", "general"
    }


def test_non_admin_manager_does_not_get_admin_level_access():
    """
    Confirms role-based override is admin-only — manager, despite having
    upload rights, should NOT get the same blanket access as admin.
    """
    user = make_user(Department.marketing, UserRole.manager)
    assert user.allowed_collections == ["marketing", "general"]
    assert set(user.allowed_collections) != {
        "engineering", "hr", "finance", "marketing", "general"
    }


# ---------------------------------------------------------------------------
# 3. primary_collection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("department", list(Department))
def test_primary_collection_matches_department_value(department):
    user = make_user(department, UserRole.viewer)
    assert user.primary_collection == department.value


# ---------------------------------------------------------------------------
# 4. can_upload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,expected", [
    (UserRole.admin,   True),
    (UserRole.manager, True),
    (UserRole.analyst, False),
    (UserRole.viewer,  False),
])
def test_can_upload_permission_by_role(role, expected):
    user = make_user(Department.engineering, role)
    assert user.can_upload is expected


# ---------------------------------------------------------------------------
# 5. can_view_logs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,expected", [
    (UserRole.admin,   True),
    (UserRole.manager, True),
    (UserRole.analyst, True),
    (UserRole.viewer,  False),
])
def test_can_view_logs_permission_by_role(role, expected):
    user = make_user(Department.finance, role)
    assert user.can_view_logs is expected


def test_viewer_cannot_upload_or_view_logs():
    """
    Explicit lowest-privilege check — a viewer should be locked out
    of both upload and log-viewing, confirming the floor of the
    permission model is correctly restrictive.
    """
    user = make_user(Department.general, UserRole.viewer)
    assert user.can_upload is False
    assert user.can_view_logs is False


# ---------------------------------------------------------------------------
# 6. Enum sanity — string-backed enums used in DB columns and API payloads
# ---------------------------------------------------------------------------

def test_department_enum_values_are_lowercase_strings():
    """
    These values get serialized into JWTs, DB rows, and JSON responses —
    confirms consistent casing so lookups (e.g. dict access by string)
    don't silently fail on a case mismatch.
    """
    for dept in Department:
        assert dept.value == dept.value.lower()


def test_userrole_enum_values_are_lowercase_strings():
    for role in UserRole:
        assert role.value == role.value.lower()


def test_department_enum_has_exactly_six_members():
    """
    Locks in the current department count — if someone adds a 7th
    department, this test fails as a reminder to also update
    DEPARTMENT_ACCESS and any department-count assumptions elsewhere.
    """
    assert len(list(Department)) == 6


def test_department_string_value_roundtrips_to_enum():
    """
    Confirms Department('finance') works — this is how a department
    string coming from a JWT claim or DB row gets turned back into
    the enum for allowed_collections lookups.
    """
    assert Department("finance") == Department.finance
    assert Department("c_level") == Department.c_level


def test_invalid_department_string_raises_valueerror():
    with pytest.raises(ValueError):
        Department("not_a_real_department")


def test_userrole_string_value_roundtrips_to_enum():
    assert UserRole("admin") == UserRole.admin
    assert UserRole("viewer") == UserRole.viewer


def test_invalid_role_string_raises_valueerror():
    with pytest.raises(ValueError):
        UserRole("superuser")