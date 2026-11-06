-- =========================================================================
-- NBA Probability Assistant - Staging Database Schema (SQL Server 2022/LocalDB)
-- Database: NBA_Prediction
-- =========================================================================

IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'NBA_Prediction')
BEGIN
    CREATE DATABASE NBA_Prediction;
END;
GO

USE NBA_Prediction;
GO

-- 1. Active Players Reference Table
IF OBJECT_ID('dbo.stg_players', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_players (
        PERSON_ID INT NOT NULL PRIMARY KEY,
        DISPLAY_FIRST_LAST NVARCHAR(100),
        DISPLAY_LAST_COMMA_FIRST NVARCHAR(100),
        ROSTERSTATUS INT,
        FROM_YEAR INT,
        TO_YEAR INT,
        PLAYERCODE NVARCHAR(50),
        TEAM_ID INT,
        TEAM_CITY NVARCHAR(50),
        TEAM_NAME NVARCHAR(50),
        TEAM_ABBREVIATION NVARCHAR(10),
        TEAM_CODE NVARCHAR(50),
        GAMES_PLAYED_FLAG NVARCHAR(5),
        OTHERLEAGUE_EXPERIENCE_CH INT,
        ingested_at DATETIME2 DEFAULT SYSDATETIME()
    );
END;
GO

-- 2. Team Game Logs (Authoritative 2014-Present)
IF OBJECT_ID('dbo.stg_team_game_logs', 'U') IS NOT NULL
    DROP TABLE dbo.stg_team_game_logs;
GO

CREATE TABLE dbo.stg_team_game_logs (
    SEASON_ID NVARCHAR(20),
    TEAM_ID INT NOT NULL,
    TEAM_ABBREVIATION NVARCHAR(10),
    TEAM_NAME NVARCHAR(100),
    GAME_ID NVARCHAR(20) NOT NULL,
    GAME_DATE NVARCHAR(20) NOT NULL,
    MATCHUP NVARCHAR(50),
    WL NVARCHAR(5),
    MIN FLOAT,
    FGM FLOAT,
    FGA FLOAT,
    FG_PCT FLOAT,
    FG3M FLOAT,
    FG3A FLOAT,
    FG3_PCT FLOAT,
    FTM FLOAT,
    FTA FLOAT,
    FT_PCT FLOAT,
    OREB FLOAT,
    DREB FLOAT,
    REB FLOAT,
    AST FLOAT,
    STL FLOAT,
    BLK FLOAT,
    TOV FLOAT,
    PF FLOAT,
    PTS FLOAT,
    PLUS_MINUS FLOAT,
    VIDEO_AVAILABLE INT,
    SEASON NVARCHAR(20) NOT NULL,
    downloaded_at NVARCHAR(50),
    ingested_at DATETIME2 DEFAULT SYSDATETIME(),
    CONSTRAINT PK_stg_team_game_logs PRIMARY KEY (TEAM_ID, GAME_ID)
);
CREATE NONCLUSTERED INDEX IX_team_game_logs_season ON dbo.stg_team_game_logs (SEASON);
CREATE NONCLUSTERED INDEX IX_team_game_logs_date ON dbo.stg_team_game_logs (GAME_DATE);
GO

-- 3. Player Game Logs (Authoritative 2014-Present)
IF OBJECT_ID('dbo.stg_player_game_logs', 'U') IS NOT NULL
    DROP TABLE dbo.stg_player_game_logs;
GO

CREATE TABLE dbo.stg_player_game_logs (
    SEASON_ID NVARCHAR(20),
    PLAYER_ID INT NOT NULL,
    PLAYER_NAME NVARCHAR(100),
    TEAM_ID INT NOT NULL,
    TEAM_ABBREVIATION NVARCHAR(10),
    TEAM_NAME NVARCHAR(100),
    GAME_ID NVARCHAR(20) NOT NULL,
    GAME_DATE NVARCHAR(20) NOT NULL,
    MATCHUP NVARCHAR(50),
    WL NVARCHAR(5),
    MIN FLOAT,
    FGM FLOAT,
    FGA FLOAT,
    FG_PCT FLOAT,
    FG3M FLOAT,
    FG3A FLOAT,
    FG3_PCT FLOAT,
    FTM FLOAT,
    FTA FLOAT,
    FT_PCT FLOAT,
    OREB FLOAT,
    DREB FLOAT,
    REB FLOAT,
    AST FLOAT,
    STL FLOAT,
    BLK FLOAT,
    TOV FLOAT,
    PF FLOAT,
    PTS FLOAT,
    PLUS_MINUS FLOAT,
    FANTASY_PTS FLOAT,
    VIDEO_AVAILABLE INT,
    SEASON NVARCHAR(20) NOT NULL,
    downloaded_at NVARCHAR(50),
    ingested_at DATETIME2 DEFAULT SYSDATETIME(),
    CONSTRAINT PK_stg_player_game_logs PRIMARY KEY (PLAYER_ID, GAME_ID)
);
CREATE NONCLUSTERED INDEX IX_player_game_logs_season ON dbo.stg_player_game_logs (SEASON);
CREATE NONCLUSTERED INDEX IX_player_game_logs_player ON dbo.stg_player_game_logs (PLAYER_ID);
CREATE NONCLUSTERED INDEX IX_player_game_logs_date ON dbo.stg_player_game_logs (GAME_DATE);
GO
