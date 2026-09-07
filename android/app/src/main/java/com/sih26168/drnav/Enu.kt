package com.sih26168.drnav

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * WGS-84 geodetic to local ENU tangent plane.
 *
 * The conversion is done on the phone so the CSV the app writes is already in
 * the same metric frame the offline pipeline works in -- `drnav.log_format`
 * reads these files directly with no adaptation step. The origin is the first
 * fix of the recording and is stored in `meta.json`.
 */
object Enu {
    private const val A = 6378137.0
    private const val F = 1.0 / 298.257223563
    private const val E2 = F * (2 - F)

    private fun toEcef(latRad: Double, lonRad: Double, alt: Double): DoubleArray {
        val s = sin(latRad)
        val n = A / sqrt(1 - E2 * s * s)
        return doubleArrayOf(
            (n + alt) * cos(latRad) * cos(lonRad),
            (n + alt) * cos(latRad) * sin(lonRad),
            (n * (1 - E2) + alt) * s,
        )
    }

    /** Returns east, north, up in metres relative to (lat0, lon0, alt0). */
    fun fromGeodetic(
        lat: Double, lon: Double, alt: Double,
        lat0: Double, lon0: Double, alt0: Double,
    ): DoubleArray {
        val latR = Math.toRadians(lat)
        val lonR = Math.toRadians(lon)
        val lat0R = Math.toRadians(lat0)
        val lon0R = Math.toRadians(lon0)

        val p = toEcef(latR, lonR, alt)
        val p0 = toEcef(lat0R, lon0R, alt0)
        val dx = p[0] - p0[0]
        val dy = p[1] - p0[1]
        val dz = p[2] - p0[2]

        val sl = sin(lat0R); val cl = cos(lat0R)
        val so = sin(lon0R); val co = cos(lon0R)
        return doubleArrayOf(
            -so * dx + co * dy,
            -sl * co * dx - sl * so * dy + cl * dz,
            cl * co * dx + cl * so * dy + sl * dz,
        )
    }
}
